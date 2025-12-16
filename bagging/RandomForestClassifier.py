import pandas as pd
import numpy as np
import time
import psutil
import joblib

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


from module_metrics import compute_all_metrics
from module_roc import plot_multiclass_roc
from module_confusion_matrix import plot_conf_matrix


# -------------------------------------------------------------------
# Step 1: Load Dataset
# -------------------------------------------------------------------
data = pd.read_csv("load_test_dataset.csv")
print("Dataset Loaded...")

data = data[data['ResponseCode'] == 200].reset_index(drop=True)

def time_to_seconds(t):
    h, m, s = map(int, t.split(':'))
    return h * 3600 + m * 60 + s

data["TimeSeconds"] = data["TimeStamp"].apply(time_to_seconds)
data = data.sort_values(by="TimeSeconds").reset_index(drop=True)


# -------------------------------------------------------------------
# Step 2: Sliding Window Feature Extraction
# -------------------------------------------------------------------
def create_sliding_features(df, window_size=30, step=1):
    X, y = [], []
    for i in range(0, len(df) - window_size, step):
        window = df.iloc[i:i+window_size]
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


X_df, y = create_sliding_features(data, window_size=30, step=1)

# -------------------------------------------------------------------
# Step 3: Scaling / Encoding
# -------------------------------------------------------------------
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_df), columns=X_df.columns)

unique_labels = sorted(np.unique(y))
label_map = {label: i for i, label in enumerate(unique_labels)}
reverse_map = {i: label for label, i in label_map.items()}

y_encoded = np.array([label_map[i] for i in y])


# -------------------------------------------------------------------
# Step 4: Train/Test Split
# -------------------------------------------------------------------
X_train_all, X_test, y_train_all, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)


# -------------------------------------------------------------------
# Step 5: Training with K-Fold + Resource Monitoring
# -------------------------------------------------------------------

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cpu_usage_train = []
ram_usage_train = []
accuracies = []

train_start = time.time()

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_all, y_train_all), 1):

    print(f" Training Fold {fold}...")

    X_train = X_train_all.iloc[train_idx]
    y_train = y_train_all[train_idx]
    X_val = X_train_all.iloc[val_idx]
    y_val = y_train_all[val_idx]

    cpu_usage_train.append(psutil.cpu_percent())
    ram_usage_train.append(psutil.virtual_memory().percent)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        max_features="sqrt",
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    accuracies.append(accuracy_score(y_val, y_val_pred))

train_end = time.time()

training_time_minutes = (train_end - train_start) / 60

print(" All Folds Completed Successfully.")


# -------------------------------------------------------------------
# Step 6: Final Model Training
# -------------------------------------------------------------------
final_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    max_features="sqrt",
    random_state=42,
    class_weight="balanced"
)
final_model.fit(X_train_all, y_train_all)

joblib.dump(final_model, "rf_model_sliding.pkl")
joblib.dump(scaler, "scaler_sliding.pkl")


# -------------------------------------------------------------------
# Step 7: Prediction + Resource Monitoring
# -------------------------------------------------------------------
pred_start = time.time()
cpu_pred = psutil.cpu_percent()
ram_pred = psutil.virtual_memory().percent

y_test_pred = final_model.predict(X_test)
y_test_proba = final_model.predict_proba(X_test)

pred_end = time.time()
prediction_time_sec = pred_end - pred_start


# -------------------------------------------------------------------
# Step 8: Compute All Metrics (Using Module)
# -------------------------------------------------------------------
class_names = [reverse_map[i] for i in sorted(reverse_map)]
print("   Validation Metrics")
metrics = compute_all_metrics(
    y_true=y_test,
    y_pred=y_test_pred,
    y_proba=y_test_proba
)


print(metrics)


# -------------------------------------------------------------------
# Step 9: Plot ROC (Using Module)
# -------------------------------------------------------------------
plot_multiclass_roc(y_test, y_test_proba, class_names)


# -------------------------------------------------------------------
# Step 10: Confusion Matrix
# -------------------------------------------------------------------
plot_conf_matrix(y_test, y_test_pred, class_names)


# -------------------------------------------------------------------
# Step 11: Print Time and Resource Tables
# -------------------------------------------------------------------
print("\n=== Training / Prediction Time ===")
print(pd.DataFrame([[
    round(training_time_minutes, 2),
    round(prediction_time_sec, 3)
]], columns=["TrainingTime(min)", "PredictionTime(sec)"]))

print("\n=== CPU / RAM Usage ===")
print(pd.DataFrame([
    ["Training", np.mean(cpu_usage_train), np.mean(ram_usage_train)],
    ["Prediction", cpu_pred, ram_pred]
], columns=["Phase", "CPU(%)", "RAM(%)"]))
print("\nSaved Files:")
print(" - bagging_results.csv")
print(" - bagging_time.csv")
print(" - bagging_resources.csv")
print("\nProcess Finished Successfully.\n")
