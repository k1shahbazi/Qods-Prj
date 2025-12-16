import pandas as pd
import numpy as np
import time
import psutil
import joblib

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from collections import defaultdict

from module_metrics import compute_all_metrics
from module_roc import plot_multiclass_roc
from module_confusion_matrix import plot_conf_matrix

# ----------------------------- Config / Globals -----------------------------
DATA_PATH = "load_test_dataset.csv"
MODEL_OUT = "xgb_model_integrated.pkl"
SCALER_OUT = "scaler_integrated.pkl"
FEATURES_TO_SCALE = [
    'Elapsed', 'Elapsed_RollingMean', 'Elapsed_RollingStd',
    'Elapsed_Derivative', 'Elapsed_SecondDerivative',
    'Elapsed_Lag1', 'Elapsed_Lag2',
    'Elapsed_RollingRange', 'Elapsed_StdMeanRatio'
]

# ----------------------------- Helpers -----------------------------
def time_to_seconds(t):
    h, m, s = map(int, t.split(':'))
    return h * 3600 + m * 60 + s

# ----------------------------- Load & Preprocess -----------------------------
print("Loading data...")
data = pd.read_csv(DATA_PATH)
data = data[data['ResponseCode'] == 200].reset_index(drop=True)
data['TimeSeconds'] = data['TimeStamp'].apply(time_to_seconds)
data = data.sort_values(by='TimeSeconds').reset_index(drop=True)

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


# Normalization
scaler = StandardScaler()
data_scaled = data.copy()
data_scaled[FEATURES_TO_SCALE] = scaler.fit_transform(data_scaled[FEATURES_TO_SCALE])

# Encode labels
unique_labels = sorted(data_scaled['Label'].unique())
label_map = {label: idx for idx, label in enumerate(unique_labels)}
reverse_label_map = {v: k for k, v in label_map.items()}
data_scaled['LabelEncoded'] = data_scaled['Label'].map(label_map)

def add_controlled_noise(df, features, target_class='normal', noise_level=0.02):
    mask = df['Label'] == target_class
    if mask.sum() == 0:
        return df
    noise = np.random.normal(0, noise_level, size=(mask.sum(), len(features)))
    df.loc[mask, features] = df.loc[mask, features].values + noise
    return df

def create_mixed_samples(df, features, target_class='normal', n_samples=800, alpha_range=(0.5, 0.8)):
    if target_class not in df['Label'].values:
        return df
    normal_samples = df[df['Label'] == target_class][features]
    anomaly_samples = df[df['Label'] != target_class][features]
    if len(normal_samples) == 0 or len(anomaly_samples) == 0:
        return df
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

data_scaled = add_controlled_noise(data_scaled, FEATURES_TO_SCALE)
data_scaled = create_mixed_samples(data_scaled, FEATURES_TO_SCALE)

# Train/test split
train_data, test_data = train_test_split(
    data_scaled, test_size=0.2, stratify=data_scaled['LabelEncoded'], random_state=42
)

X_train_all = train_data[FEATURES_TO_SCALE].reset_index(drop=True)
y_train_all = train_data['LabelEncoded'].reset_index(drop=True)
X_test = test_data[FEATURES_TO_SCALE].reset_index(drop=True)
y_test = test_data['LabelEncoded'].reset_index(drop=True)

classes = np.unique(y_train_all)
weights = compute_class_weight('balanced', classes=classes, y=y_train_all)
class_weights = dict(zip(classes, weights))

# ----------------------------- Train with CV & monitor resources -----------------------------
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = []
all_reports = []
train_reports = []
accuracies = []

training_cpu_usage = []
training_ram_usage = []

training_start = time.time()
for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_all, y_train_all), 1):
    print(f" Fold {fold} / 5 ...")
    # sample current CPU/RAM
    training_cpu_usage.append(psutil.cpu_percent(interval=None))
    training_ram_usage.append(psutil.virtual_memory().percent)

    X_tr = X_train_all.iloc[train_idx]
    y_tr = y_train_all.iloc[train_idx]
    X_val = X_train_all.iloc[val_idx]
    y_val = y_train_all.iloc[val_idx]

    model = XGBClassifier(
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

    sample_weights = np.array([class_weights[int(y)] for y in y_tr])
    model.fit(X_tr, y_tr, sample_weight=sample_weights)
    models.append(model)

    y_val_pred = model.predict(X_val)
    y_train_pred = model.predict(X_tr)

    val_report = classification_report(
        y_val, y_val_pred,
        target_names=[reverse_label_map[i] for i in sorted(reverse_label_map)],
        output_dict=True
    )
    train_report = classification_report(
        y_tr, y_train_pred,
        target_names=[reverse_label_map[i] for i in sorted(reverse_label_map)],
        output_dict=True
    )

    all_reports.append(val_report)
    train_reports.append(train_report)
    accuracies.append(accuracy_score(y_val, y_val_pred))

training_end = time.time()
training_time_minutes = (training_end - training_start) / 60.0
print("     All Folds Completed Successfully!")

# ----------------------------- Average CV metrics -----------------------------
print("   Validation Metrics")

avg_val_report = defaultdict(lambda: defaultdict(float))
avg_train_report = defaultdict(lambda: defaultdict(float))
class_names = [reverse_label_map[i] for i in sorted(reverse_label_map)]

for report in all_reports:
    for label in class_names:
        for metric in ['precision', 'recall', 'f1-score']:
            avg_val_report[label][metric] += report[label][metric] / len(all_reports)

for report in train_reports:
    for label in class_names:
        for metric in ['precision', 'recall', 'f1-score']:
            avg_train_report[label][metric] += report[label][metric] / len(train_reports)

val_overall_metrics = {metric: np.mean([avg_val_report[label][metric] for label in class_names]) for metric in ['precision', 'recall', 'f1-score']}
train_overall_metrics = {metric: np.mean([avg_train_report[label][metric] for label in class_names]) for metric in ['precision', 'recall', 'f1-score']}

print(f" {np.mean(accuracies):.4f}")
for m, v in val_overall_metrics.items():
    print(f" {m.capitalize()}: {v:.4f}")

print(f" {accuracy_score(y_train_all, models[-1].predict(X_train_all)):.4f}")
for m, v in train_overall_metrics.items():
    print(f" {m.capitalize()}: {v:.4f}")

# ----------------------------- Final model train on all train data -----------------------------
final_model = models[-1]
final_model.fit(X_train_all, y_train_all)
joblib.dump(final_model, MODEL_OUT)
joblib.dump(scaler, SCALER_OUT)

# ----------------------------- Prediction + resources -----------------------------
pred_start = time.time()
y_test_pred = final_model.predict(X_test)

y_test_proba = None
try:
    y_test_proba = final_model.predict_proba(X_test)
except Exception:
    y_test_proba = None

pred_end = time.time()
prediction_time_sec = pred_end - pred_start

prediction_cpu = psutil.cpu_percent(interval=None)
prediction_ram = psutil.virtual_memory().percent

# ----------------------------- Compute final metrics with module -----------------------------
metrics = compute_all_metrics(y_true=y_test.values, y_pred=y_test_pred, y_proba=y_test_proba if y_test_proba is not None else np.zeros((len(y_test), len(class_names))))

print(f" {training_time_minutes:.2f}")
print(f" {prediction_time_sec:.4f}")
print(f" {np.mean(training_cpu_usage):.2f}")
print(f" {np.mean(training_ram_usage):.2f}")
print(f" {prediction_cpu:.2f}")
print(f" {prediction_ram:.2f}")
print("   Validation Metrics")

for k, v in metrics.items():
    if v is None:
        print(f" {k}: None")
    else:
        print(f" {k}: {v:.4f}")

# Save metrics/time/resources to CSVs
metrics_df = pd.DataFrame([metrics])
time_df = pd.DataFrame([{
    "training_time_min": training_time_minutes,
    "prediction_time_sec": prediction_time_sec
}])
resources_df = pd.DataFrame([
    {"phase": "training", "cpu_avg_percent": np.mean(training_cpu_usage), "ram_avg_percent": np.mean(training_ram_usage)},
    {"phase": "prediction", "cpu_percent": prediction_cpu, "ram_percent": prediction_ram}
])

metrics_df.to_csv("Boosting_results.csv.csv", index=False)
time_df.to_csv("Boosting_time.csv", index=False)
resources_df.to_csv("Boostinging_resources.csv", index=False)

# ----------------------------- Plots: ROC + Confusion Matrix -----------------------------
if y_test_proba is not None:
    plot_multiclass_roc(y_test.values, y_test_proba, class_names)
else:
    print(" Probability estimates not available — skipping ROC plot.")

plot_conf_matrix(y_test.values, y_test_pred, class_names)


print("\nSaved Files:")
print(" - Boosting_results.csv")
print(" - Boosting_time.csv")
print(" - Boosting_resources.csv")

print(f" - {MODEL_OUT}")
print("\nProcess finished.")
