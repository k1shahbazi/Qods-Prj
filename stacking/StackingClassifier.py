import time
import numpy as np
import pandas as pd
import psutil
import joblib
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.base import clone
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from module_metrics import compute_all_metrics
from module_roc import plot_multiclass_roc
from module_confusion_matrix import plot_conf_matrix
from module_resources import measure_training_resources, monitor_prediction

# ----------------- Config -----------------
DATA_PATH = "load_test_dataset.csv"
MODEL_OUT = "stacked_model_light.pkl"
SCALER_OUT = "scaler_light.pkl"
WINDOW_SIZE = 30
FEATURES_TO_SCALE = [
    'Elapsed', 'Elapsed_RollingMean', 'Elapsed_RollingStd',
    'Elapsed_Derivative', 'Elapsed_SecondDerivative',
    'Elapsed_Lag1', 'Elapsed_Lag2',
    'Elapsed_RollingRange', 'Elapsed_StdMeanRatio'
]

# ----------------- Helpers -----------------
def time_to_seconds(t):
    h, m, s = map(int, t.split(':'))
    return h * 3600 + m * 60 + s

def create_sliding_features(df, window_size=30, step=1):
    X, y = [], []
    for i in range(0, len(df) - window_size, step):
        window = df.iloc[i:i + window_size]
        feats = {
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
        X.append(feats)
        y.append(window['Label'].iloc[-1])
    return pd.DataFrame(X), np.array(y)

# ----------------- Load & preprocess -----------------
print("Loading data...")
data = pd.read_csv(DATA_PATH)
data = data[data['ResponseCode'] == 200].reset_index(drop=True)
data['TimeSeconds'] = data['TimeStamp'].apply(time_to_seconds)
data = data.sort_values(by='TimeSeconds').reset_index(drop=True)

data['Elapsed_RollingMean'] = data['Elapsed'].rolling(window=WINDOW_SIZE, min_periods=1).mean()
data['Elapsed_RollingStd'] = data['Elapsed'].rolling(window=WINDOW_SIZE, min_periods=1).std().fillna(0)
data['Elapsed_Derivative'] = data['Elapsed'].diff().fillna(0)
data['Elapsed_SecondDerivative'] = data['Elapsed_Derivative'].diff().fillna(0)
data['Elapsed_Lag1'] = data['Elapsed'].shift(1).bfill()
data['Elapsed_Lag2'] = data['Elapsed'].shift(2).bfill()
data['Elapsed_RollingRange'] = data['Elapsed'].rolling(window=WINDOW_SIZE, min_periods=1).apply(lambda x: x.max() - x.min(), raw=True)
data['Elapsed_StdMeanRatio'] = data['Elapsed_RollingStd'] / (data['Elapsed_RollingMean'] + 1e-6)

X_df, y_series = create_sliding_features(data, window_size=WINDOW_SIZE, step=1)

# scale features
scaler = StandardScaler()
data_scaled = data.copy()
data_scaled[FEATURES_TO_SCALE] = scaler.fit_transform(data_scaled[FEATURES_TO_SCALE])

# encode labels
unique_labels = sorted(data_scaled['Label'].unique())
label_map = {label: idx for idx, label in enumerate(unique_labels)}
reverse_label_map = {v: k for k, v in label_map.items()}
data_scaled['LabelEncoded'] = data_scaled['Label'].map(label_map)

def add_controlled_noise(df, features, target_class='normal', noise_level=0.03):
    mask = df['Label'] == target_class
    if mask.sum() == 0:
        return df
    noise = np.random.normal(0, noise_level, size=(mask.sum(), len(features)))
    df.loc[mask, features] = df.loc[mask, features].values + noise
    return df

def create_mixed_samples(df, features, target_class='normal', n_samples=400, alpha_range=(0.6, 0.8)):
    if target_class not in df['Label'].values:
        return df
    normal_samples = df[df['Label'] == target_class][features]
    anomaly_samples = df[df['Label'] != target_class][features]
    if len(normal_samples)==0 or len(anomaly_samples)==0:
        return df
    mixed_samples = []
    for _ in range(n_samples):
        alpha = np.random.uniform(*alpha_range)
        normal_sample = normal_samples.sample(1).values[0]
        anomaly_sample = anomaly_samples.sample(1).values[0]
        mixed = alpha*normal_sample + (1-alpha)*anomaly_sample
        mixed_samples.append(mixed)
    mixed_df = pd.DataFrame(mixed_samples, columns=features)
    mixed_df['Label'] = target_class
    mixed_df['LabelEncoded'] = label_map[target_class]
    return pd.concat([df, mixed_df], ignore_index=True)

data_scaled = add_controlled_noise(data_scaled, FEATURES_TO_SCALE)
data_scaled = create_mixed_samples(data_scaled, FEATURES_TO_SCALE)

# train / test split
train_data, test_data = train_test_split(data_scaled, test_size=0.2, stratify=data_scaled['LabelEncoded'], random_state=42)
X_train_all = train_data[FEATURES_TO_SCALE].reset_index(drop=True)
y_train_all = train_data['LabelEncoded'].reset_index(drop=True)
X_test = test_data[FEATURES_TO_SCALE].reset_index(drop=True)
y_test = test_data['LabelEncoded'].reset_index(drop=True)

classes = np.unique(y_train_all)
weights = compute_class_weight('balanced', classes=classes, y=y_train_all)
class_weights = dict(zip(classes, weights))

class_names = [reverse_label_map[i] for i in sorted(reverse_label_map)]
n_classes = len(class_names)

# ----------------- Define base learners -----------------
rf_model = RandomForestClassifier(
    n_estimators=150, max_depth=8, min_samples_split=10, min_samples_leaf=5,
    max_features=0.6, class_weight=class_weights, random_state=42, bootstrap=True
)
xgb_model = XGBClassifier(
    n_estimators=150, max_depth=6, learning_rate=0.1, gamma=0.3, min_child_weight=3,
    subsample=0.7, colsample_bytree=0.8, reg_alpha=0.2, reg_lambda=1.5,
    use_label_encoder=False, eval_metric='mlogloss', random_state=42, tree_method='hist'
)
knn_model = KNeighborsClassifier(n_neighbors=15, weights='distance', p=2)

# ----------------- Training function for resource wrapper (used by module_resources) -----------------
def train_cv_collect_resources(cpu_usage_list=None, ram_usage_list=None):

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = []              
    all_reports = []
    train_reports = []
    accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_all, y_train_all), 1):
        print(f" Training Fold {fold} / {kfold} ...")
        if cpu_usage_list is not None:
            cpu_usage_list.append(psutil.cpu_percent(interval=None))
        if ram_usage_list is not None:
            ram_usage_list.append(psutil.virtual_memory().percent)

        X_tr = X_train_all.iloc[train_idx]
        y_tr = y_train_all.iloc[train_idx]
        X_val = X_train_all.iloc[val_idx]
        y_val = y_train_all.iloc[val_idx]

        rf_clone = clone(rf_model)
        xgb_clone = clone(xgb_model)
        knn_clone = clone(knn_model)

        sample_weights = np.array([class_weights[int(y)] for y in y_tr])
        rf_clone.fit(X_tr, y_tr)
        xgb_clone.fit(X_tr, y_tr, sample_weight=sample_weights)
        knn_clone.fit(X_tr, y_tr)

        rf_proba = rf_clone.predict_proba(X_val)
        xgb_proba = xgb_clone.predict_proba(X_val)
        knn_proba = knn_clone.predict_proba(X_val)
        meta_X_val = np.hstack([rf_proba, xgb_proba, knn_proba])

        meta_model = LogisticRegression(max_iter=1000, C=0.1, penalty='l2', solver='lbfgs', multi_class='multinomial', random_state=42)
        meta_model.fit(meta_X_val, y_val)
        stack_pred = meta_model.predict(meta_X_val)

        val_report = classification_report(y_val, stack_pred, target_names=class_names, output_dict=True)
        all_reports.append(val_report)
        accuracies.append(accuracy_score(y_val, stack_pred))

        models.append({'rf': rf_clone, 'xgb': xgb_clone, 'knn': knn_clone, 'meta': meta_model})

        print(f"  Fold {fold} accuracy: {accuracies[-1]:.4f}")

    return {
        'models': models,
        'all_reports': all_reports,
        'accuracies': accuracies
    }
print("     All Folds Completed Successfully!")

# ----------------- Run CV training under resource monitor -----------------
train_summary = measure_training_resources(train_cv_collect_resources)
training_time_sec = train_summary.get('training_time_sec', None)
training_cpu_avg = train_summary.get('cpu_avg', None)
training_ram_avg = train_summary.get('ram_avg', None)
cv_result = train_summary.get('result', {})

# ----------------- Compute CV aggregated metrics (light) -----------------
print("   Validation Metrics")
all_reports = cv_result.get('all_reports', [])
if all_reports:
    val_acc_mean = np.mean(cv_result.get('accuracies', [0]))
    print(f"CV mean accuracy: {val_acc_mean:.4f}")

# ----------------- Train final stacking on full train set (measure resources lightly) -----------------
train_start_final = time.time()
train_cpu_before = psutil.cpu_percent(interval=None)
train_ram_before = psutil.virtual_memory().percent

rf_model.fit(X_train_all, y_train_all)
xgb_model.fit(X_train_all, y_train_all, sample_weight=np.array([class_weights[int(y)] for y in y_train_all]))
knn_model.fit(X_train_all, y_train_all)

rf_proba_train = rf_model.predict_proba(X_train_all)
xgb_proba_train = xgb_model.predict_proba(X_train_all)
knn_proba_train = knn_model.predict_proba(X_train_all)
meta_features_train = np.hstack([rf_proba_train, xgb_proba_train, knn_proba_train])

final_meta_model = LogisticRegression(max_iter=1000, C=0.1, penalty='l2', solver='lbfgs', multi_class='multinomial', random_state=42)
final_meta_model.fit(meta_features_train, y_train_all)

train_end_final = time.time()
train_cpu_after = psutil.cpu_percent(interval=None)
train_ram_after = psutil.virtual_memory().percent

final_training_time_sec = train_end_final - train_start_final
final_train_cpu_avg = np.mean([train_cpu_before, train_cpu_after])
final_train_ram_avg = np.mean([train_ram_before, train_ram_after])

stacked_model = {
    'base_models': {'random_forest': rf_model, 'xgboost': xgb_model, 'knn': knn_model},
    'meta_model': final_meta_model,
    'n_classes': n_classes,
    'class_names': class_names
}

joblib.dump(stacked_model, MODEL_OUT)
joblib.dump(scaler, SCALER_OUT)
# ----------------- Prediction (wrapped by module_resources.monitor_prediction) -----------------
def do_prediction():
    
    rf_p = stacked_model['base_models']['random_forest'].predict_proba(X_test)
    xgb_p = stacked_model['base_models']['xgboost'].predict_proba(X_test)
    knn_p = stacked_model['base_models']['knn'].predict_proba(X_test)
    meta_X_test = np.hstack([rf_p, xgb_p, knn_p])

    y_test_pred = stacked_model['meta_model'].predict(meta_X_test)
    y_test_proba = None
    try:
        y_test_proba = stacked_model['meta_model'].predict_proba(meta_X_test)
    except Exception:
        y_test_proba = None

    return {'y_pred': y_test_pred, 'y_proba': y_test_proba}

pred_summary = monitor_prediction(do_prediction)
prediction_time_sec = pred_summary.get('prediction_time_sec', None)
prediction_cpu = pred_summary.get('cpu', None)
prediction_ram = pred_summary.get('ram', None)
pred_result = pred_summary.get('result', {})

y_test_pred = pred_result.get('y_pred', None)
y_test_proba = pred_result.get('y_proba', None)

# ----------------- Compute & print final metrics (use module_metrics) -----------------
metrics = compute_all_metrics(y_true=y_test.values if hasattr(y_test, "values") else y_test,
                              y_pred=y_test_pred,
                              y_proba=y_test_proba if y_test_proba is not None else np.zeros((len(y_test), n_classes)))
print("\nTraining / Prediction durations:")
print(f" {training_time_sec}")
print(f" {final_training_time_sec:.3f}")
print(f" {prediction_time_sec:.4f}" if prediction_time_sec is not None else "Prediction time: N/A")

print(f" {training_cpu_avg}")
print(f" {training_ram_avg}")
print(f" {final_train_cpu_avg:.2f}")
print(f" {final_train_ram_avg:.2f}")
print(f" {prediction_cpu}")
print(f" {prediction_ram}")

for k, v in metrics.items():
    print(f" {k}: {v}")

# save predictions for review
test_out = test_data.copy()
test_out['Predicted_LabelEncoded'] = y_test_pred
test_out['Predicted_Label'] = test_out['Predicted_LabelEncoded'].map(reverse_label_map)
test_out.to_csv("Stacking_results.csv", index=False)


# ----------------- Plots -----------------
if y_test_proba is not None:
    plot_multiclass_roc(y_test.values if hasattr(y_test, "values") else y_test, y_test_proba, class_names)
else:
    print("No probabilities for meta-model -> ROC skipped.")

plot_conf_matrix(y_test.values if hasattr(y_test, "values") else y_test, y_test_pred, class_names)

print("\nSaved Files:")
print(" - Stacking_results.csv")
print(" - Stacking_time.csv")
print(" - Stacking_resources.csv")
print("\nProcess Finished.\n")
