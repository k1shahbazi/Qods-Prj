import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (classification_report, accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.base import clone
import joblib
import seaborn as sns
from collections import defaultdict

# --------------- Step 1: Read and Prepare Dataset ---------------
print("Loading and preprocessing data...")
data = pd.read_csv('load_test_dataset.csv')

data = data[data['ResponseCode'] == 200].reset_index(drop=True)

def time_to_seconds(t):
    h, m, s = map(int, t.split(':'))
    return h * 3600 + m * 60 + s

data['TimeSeconds'] = data['TimeStamp'].apply(time_to_seconds)
data = data.sort_values(by='TimeSeconds').reset_index(drop=True)

# --------------- Step 2: Feature Engineering ---------------
print("Creating features...")
def create_sliding_features(df, window_size=30, step=1):
    X, y = [], []
    for i in range(0, len(df) - window_size, step):
        window = df.iloc[i:i + window_size]
        features = {
            'Elapsed_mean': window['Elapsed'].mean(),
            'Elapsed_std': window['Elapsed'].std(),
            'Elapsed_min': window['Elapsed'].min(),
            'Elapsed_max': window['Elapsed'].max(),
            'Elapsed_range': window['Elapsed'].max() - window['Elapsed'].min(),
            'Elapsed_skew': window['Elapsed'].skew(),
            'Elapsed_kurt': window['Elapsed'].kurt(),
            'Elapsed_derivative_mean': window['Elapsed'].diff().mean(),
            'Elapsed_derivative_std': window['Elapsed'].diff().std(),
        }
        X.append(features)
        y.append(window['Label'].iloc[-1])
    return pd.DataFrame(X), np.array(y)

window_size = 30
step_size = 1
X_df, y = create_sliding_features(data, window_size=window_size, step=step_size)

data['Elapsed_RollingMean'] = data['Elapsed'].rolling(window=window_size, min_periods=1).mean()
data['Elapsed_RollingStd'] = data['Elapsed'].rolling(window=window_size, min_periods=1).std().fillna(0)
data['Elapsed_Derivative'] = data['Elapsed'].diff().fillna(0)
data['Elapsed_SecondDerivative'] = data['Elapsed_Derivative'].diff().fillna(0)
data['Elapsed_Lag1'] = data['Elapsed'].shift(1).bfill()
data['Elapsed_Lag2'] = data['Elapsed'].shift(2).bfill()
data['Elapsed_RollingRange'] = data['Elapsed'].rolling(window=window_size, min_periods=1).apply(lambda x: x.max() - x.min(), raw=True)
data['Elapsed_StdMeanRatio'] = data['Elapsed_RollingStd'] / (data['Elapsed_RollingMean'] + 1e-6)

# --------------- Step 3: Normalization ---------------
print("Normalizing features...")
features_to_scale = ['Elapsed', 'Elapsed_RollingMean', 'Elapsed_RollingStd',
                     'Elapsed_Derivative', 'Elapsed_SecondDerivative',
                     'Elapsed_Lag1', 'Elapsed_Lag2',
                     'Elapsed_RollingRange', 'Elapsed_StdMeanRatio']

scaler = StandardScaler()
data_scaled = data.copy()
data_scaled[features_to_scale] = scaler.fit_transform(data_scaled[features_to_scale])

# --------------- Step 4: Encode Labels ---------------
print("Encoding labels...")
unique_labels = sorted(data_scaled['Label'].unique())
label_map = {label: idx for idx, label in enumerate(unique_labels)}
reverse_label_map = {v: k for k, v in label_map.items()}
data_scaled['LabelEncoded'] = data_scaled['Label'].map(label_map)

# --------------- Step 5: Data Augmentation ---------------
print("Augmenting data...")
def add_controlled_noise(df, features, target_class='normal', noise_level=0.03):
    mask = df['Label'] == target_class
    noise = np.random.normal(0, noise_level, size=(mask.sum(), len(features)))
    df.loc[mask, features] = df.loc[mask, features].values + noise
    return df

def create_mixed_samples(df, features, target_class='normal', n_samples=800, alpha_range=(0.6, 0.8)):
    normal_samples = df[df['Label'] == target_class][features]
    anomaly_samples = df[df['Label'] != target_class][features]
    
    mixed_samples = []
    for _ in range(n_samples):
        alpha = np.random.uniform(*alpha_range)
        normal_sample = normal_samples.sample(1).values[0]
        anomaly_sample = anomaly_samples.sample(1).values[0]
        mixed = alpha * normal_sample + (1-alpha) * anomaly_sample
        mixed_samples.append(mixed)
    
    mixed_df = pd.DataFrame(mixed_samples, columns=features)
    mixed_df['Label'] = target_class
    mixed_df['LabelEncoded'] = label_map[target_class]
    return pd.concat([df, mixed_df], ignore_index=True)

data_scaled = add_controlled_noise(data_scaled, features_to_scale)
data_scaled = create_mixed_samples(data_scaled, features_to_scale)

# --------------- Step 6: Train/Test Split ---------------
print("Splitting data into train/test sets...")
train_data, test_data = train_test_split(
    data_scaled, test_size=0.2, stratify=data_scaled['LabelEncoded'], random_state=42
)

X_train_all = train_data[features_to_scale]
y_train_all = train_data['LabelEncoded']
X_test = test_data[features_to_scale]
y_test = test_data['LabelEncoded']

classes = np.unique(y_train_all)
weights = compute_class_weight('balanced', classes=classes, y=y_train_all)
class_weights = dict(zip(classes, weights))

# --------------- Step 7: Define Base Models with Regularization ---------------
print("Initializing base models with regularization...")

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    class_weight=class_weights,
    random_state=42,
    bootstrap=True,
    oob_score=True
)

xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        scale_pos_weight=1,
        use_label_encoder=False,
        eval_metric='mlogloss',
        tree_method='hist'
    )


knn_model = KNeighborsClassifier(
    n_neighbors=15,
    weights='distance',
    p=2,
    metric='minkowski'
)

# --------------- Step 8: Stacking Classifier with K-Fold ---------------
print("Training Stacking Classifier with 5-fold CV...")
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_reports = []
conf_matrices = []

n_classes = len(np.unique(y_train_all))
oof_predictions = np.zeros((len(X_train_all), n_classes * 3)) 

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_all, y_train_all), 1):
    print(f"\nProcessing fold {fold}...")
    X_train, X_val = X_train_all.iloc[train_idx], X_train_all.iloc[val_idx]
    y_train, y_val = y_train_all.iloc[train_idx], y_train_all.iloc[val_idx]
    
    sample_weights = np.array([class_weights[y] for y in y_train])
    
    rf_clone = clone(rf_model)
    xgb_clone = clone(xgb_model)
    knn_clone = clone(knn_model)
    
    rf_clone.fit(X_train, y_train)
    xgb_clone.fit(X_train, y_train, sample_weight=sample_weights)
    knn_clone.fit(X_train, y_train)
    
    rf_proba = rf_clone.predict_proba(X_val)
    xgb_proba = xgb_clone.predict_proba(X_val)
    knn_proba = knn_clone.predict_proba(X_val)
    
    oof_predictions[val_idx, 0*n_classes:1*n_classes] = rf_proba
    oof_predictions[val_idx, 1*n_classes:2*n_classes] = xgb_proba
    oof_predictions[val_idx, 2*n_classes:3*n_classes] = knn_proba
    
    meta_model = LogisticRegression(
        max_iter=1000,
        C=0.1,
        penalty='l2',
        solver='lbfgs',
        multi_class='multinomial',
        random_state=42
    )
    meta_model.fit(oof_predictions[val_idx], y_val)
    
    stack_pred = meta_model.predict(oof_predictions[val_idx])
    
    val_report = classification_report(
        y_val, stack_pred,
        target_names=[reverse_label_map[i] for i in sorted(reverse_label_map)],
        output_dict=True
    )
    all_reports.append(val_report)
    
    fold_acc = accuracy_score(y_val, stack_pred)
    print(f"Fold {fold} Validation Accuracy: {fold_acc:.4f}")

# --------------- Step 9: Average Metrics Across Folds ---------------
print("\nCalculating average metrics...")
def calculate_average_metrics(reports):
    avg_report = defaultdict(lambda: defaultdict(float))
    for report in reports:
        for label in class_names:
            for metric in ['precision', 'recall', 'f1-score']:
                avg_report[label][metric] += report[label][metric] / len(reports)
    return avg_report

class_names = [reverse_label_map[i] for i in sorted(reverse_label_map)]
avg_val_report = calculate_average_metrics(all_reports)

def calculate_overall_metrics(reports):
    return {
        'Accuracy': np.mean([report['accuracy'] for report in reports]),
        'Precision ': np.mean([report['macro avg']['precision'] for report in reports]),
        'Recall ': np.mean([report['macro avg']['recall'] for report in reports]),
        'F1-score ': np.mean([report['macro avg']['f1-score'] for report in reports]),
    }

val_overall = calculate_overall_metrics(all_reports)

print("\n--- Overall Validation Metrics ---")
for metric, value in val_overall.items():
    print(f"  {metric}: {value:.4f}")

# --------------- Step 10: Train Final Stacking Model ---------------
print("\nTraining final stacking model on full dataset...")

print("Training base models...")
rf_model.fit(X_train_all, y_train_all)
xgb_model.fit(X_train_all, y_train_all, 
             sample_weight=np.array([class_weights[y] for y in y_train_all]))
knn_model.fit(X_train_all, y_train_all)

print("Generating meta-features...")
rf_proba = rf_model.predict_proba(X_train_all)
xgb_proba = xgb_model.predict_proba(X_train_all)
knn_proba = knn_model.predict_proba(X_train_all)

meta_features_train = np.hstack([rf_proba, xgb_proba, knn_proba])

print("Training meta-model...")
final_meta_model = LogisticRegression(
    max_iter=1000,
    C=0.1,
    penalty='l2',
    solver='lbfgs',
    multi_class='multinomial',
    random_state=42
)
final_meta_model.fit(meta_features_train, y_train_all)

stacked_model = {
    'base_models': {
        'random_forest': rf_model,
        'xgboost': xgb_model,
        'knn': knn_model
    },
    'meta_model': final_meta_model,
    'n_classes': n_classes,
    'class_names': class_names
}

joblib.dump(stacked_model, 'stacked_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model saved successfully.")

# --------------- Step 11: Evaluate Final Model ---------------
def predict_stacking(model_dict, X):
    rf_proba = model_dict['base_models']['random_forest'].predict_proba(X)
    xgb_proba = model_dict['base_models']['xgboost'].predict_proba(X)
    knn_proba = model_dict['base_models']['knn'].predict_proba(X)
    
    meta_features = np.hstack([rf_proba, xgb_proba, knn_proba])
    return model_dict['meta_model'].predict(meta_features), model_dict['meta_model'].predict_proba(meta_features)

print("\nEvaluating final model...")
y_train_pred, y_train_prob = predict_stacking(stacked_model, X_train_all)
y_test_pred, y_test_prob = predict_stacking(stacked_model, X_test)
n_classes = len(class_names)

def get_metrics(y_true, y_pred, y_prob, label_map, set_name):
    print(f"\n===== {set_name} Metrics =====")
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision ': precision_score(y_true, y_pred, average='macro'),
        'Recall ': recall_score(y_true, y_pred, average='macro'),
        'F1-score ': f1_score(y_true, y_pred, average='macro'),
    }

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    return metrics


test_metrics = get_metrics(y_test, y_test_pred, y_test_prob,
                          reverse_label_map, "Test")

test_data['Predicted_LabelEncoded'] = y_test_pred
test_data['Predicted_Label'] = test_data['Predicted_LabelEncoded'].map(reverse_label_map)
test_data.to_csv("stacked_unseen_test_predictions.csv", index=False)

print("\n=== Stacking Model Training Complete ===")

