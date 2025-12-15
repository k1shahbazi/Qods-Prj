import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from collections import defaultdict
import joblib

# ---------------- Step 1: Read Dataset ----------------
data = pd.read_csv('load_test_dataset.csv')

# ---------------- Step 2: Preprocessing ----------------
data = data[data['ResponseCode'] == 200].reset_index(drop=True)

def time_to_seconds(t):
    h, m, s = map(int, t.split(':'))
    return h * 3600 + m * 60 + s

data['TimeSeconds'] = data['TimeStamp'].apply(time_to_seconds)
data = data.sort_values(by='TimeSeconds').reset_index(drop=True)

# ---------------- Step 3: Sliding Window Feature Extraction ----------------
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

# ---------------- Step 4: Normalization ----------------
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_df), columns=X_df.columns)

# ---------------- Step 5: Encode Labels ----------------
unique_labels = sorted(np.unique(y))
label_map = {label: idx for idx, label in enumerate(unique_labels)}
reverse_label_map = {v: k for k, v in label_map.items()}
y_encoded = np.array([label_map[label] for label in y])

# ---------------- Step 6: Train/Test Split ----------------
X_train_all, X_test, y_train_all, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# ---------------- Step 7: Stratified K-Fold Training ----------------
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_reports, train_reports, accuracies, models = [], [], [], []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_all, y_train_all), 1):
    X_train, X_val = X_train_all.iloc[train_idx], X_train_all.iloc[val_idx]
    y_train, y_val = y_train_all[train_idx], y_train_all[val_idx]

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        max_features='sqrt',
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    models.append(model)

    y_val_pred = model.predict(X_val)
    y_train_pred = model.predict(X_train)

    val_report = classification_report(
        y_val, y_val_pred,
        target_names=[reverse_label_map[i] for i in sorted(reverse_label_map)],
        output_dict=True
    )
    all_reports.append(val_report)

    train_report = classification_report(
        y_train, y_train_pred,
        target_names=[reverse_label_map[i] for i in sorted(reverse_label_map)],
        output_dict=True
    )
    train_reports.append(train_report)

    acc = accuracy_score(y_val, y_val_pred)
    accuracies.append(acc)

# ---------------- Step 8: Average Metrics ----------------
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

print("\n--- Overall Validation Metrics ---")
print(f"  Accuracy:  {np.mean(accuracies):.4f}")
for metric in val_overall_metrics:
    print(f"  {metric.capitalize()}: {val_overall_metrics[metric]:.4f}")

print("\n--- Overall Training Metrics ---")
print(f"  Accuracy:  {accuracy_score(y_train_all, models[-1].predict(X_train_all)):.4f}")
for metric in train_overall_metrics:
    print(f"  {metric.capitalize()}: {train_overall_metrics[metric]:.4f}")
# ---------------- Step 9: Final Model ----------------
final_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    max_features='sqrt',
    random_state=42,
    class_weight='balanced'
)
final_model.fit(X_train_all, y_train_all)

joblib.dump(final_model, 'rf_model_sliding.pkl')
joblib.dump(scaler, 'scaler_sliding.pkl')

y_test_proba = final_model.predict_proba(X_test)
y_train_proba = final_model.predict_proba(X_train_all)
n_classes = len(class_names)

# ---------------- Step 10: Test Evaluation ----------------
y_test_pred = final_model.predict(X_test)
test_report = classification_report(
    y_test, y_test_pred,
    target_names=[reverse_label_map[i] for i in sorted(reverse_label_map)],
    output_dict=True
)

test_overall = {
    'accuracy': accuracy_score(y_test, y_test_pred),
    'precision': np.mean([test_report[label]['precision'] for label in reverse_label_map.values()]),
    'recall': np.mean([test_report[label]['recall'] for label in reverse_label_map.values()]),
    'f1-score': np.mean([test_report[label]['f1-score'] for label in reverse_label_map.values()])
}

print("\n\n======== FINAL EVALUATION ON UNSEEN TEST SET ========")
for metric, value in test_overall.items():
    print(f"{metric.capitalize()}: {value:.4f}")

print("Final Accuracy on Unseen Test Set:", accuracy_score(y_test, y_test_pred))






