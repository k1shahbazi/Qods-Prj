import time
import numpy as np
import pandas as pd
import psutil
import joblib
from collections import defaultdict
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# base learners / meta learners
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

# helper modules (must be in same folder)
from module_metrics import compute_all_metrics
from module_roc import plot_multiclass_roc
from module_confusion_matrix import plot_conf_matrix
from module_resources import measure_training_resources, monitor_prediction

# ---------------- CONFIG ----------------
DATA_PATH = "load_test_dataset.csv"
ARTIFACT_OUT = "stacked_softvote_artifact_full.pkl"
PRED_OUT = "stacked_softvote_test_predictions_full.csv"
METRICS_OUT = "stacking_metrics_full.csv"
TIME_OUT = "stacking_time_full.csv"
RES_OUT = "stacking_resources_full.csv"

RANDOM_STATE = 42
WINDOW_SIZE = 30
STEP_SIZE = 1
TEST_SIZE = 0.2
N_SPLITS = 5

# ---------------- helpers ----------------
def time_to_seconds(t):
    h, m, s = map(int, t.split(':'))
    return h*3600 + m*60 + s

def create_sliding_features(df, window_size=30, step=1):
    X_rows, y_rows = [], []
    n = len(df)
    for start in range(0, n - window_size + 1, step):
        end = start + window_size
        window = df.iloc[start:end]
        vals = window['Elapsed'].values
        feats = {
            'Elapsed_mean': np.mean(vals),
            'Elapsed_std': np.std(vals, ddof=1) if len(vals) > 1 else 0.0,
            'Elapsed_min': np.min(vals),
            'Elapsed_max': np.max(vals),
            'Elapsed_range': np.max(vals) - np.min(vals),
            'Elapsed_skew': pd.Series(vals).skew(),
            'Elapsed_kurt': pd.Series(vals).kurt(),
            'Elapsed_derivative_mean': np.mean(np.diff(vals)) if len(vals) > 1 else 0.0,
            'Elapsed_derivative_std': np.std(np.diff(vals), ddof=1) if len(vals) > 1 else 0.0
        }
        X_rows.append(feats)
        y_rows.append(window['Label'].iloc[-1])
    return pd.DataFrame(X_rows), np.array(y_rows)

# ---------------- load & preprocess ----------------
print("   Dataset Loaded ... ")
data = pd.read_csv(DATA_PATH)
data = data[data["ResponseCode"] == 200].reset_index(drop=True)
data['TimeSeconds'] = data['TimeStamp'].apply(time_to_seconds)
data = data.sort_values(by='TimeSeconds').reset_index(drop=True)

X_df, y = create_sliding_features(data, window_size=WINDOW_SIZE, step=STEP_SIZE)
print(f"{len(X_df)}  {X_df.shape}")

unique_labels = sorted(np.unique(y))
label_map = {lab: idx for idx, lab in enumerate(unique_labels)}
reverse_label_map = {v: k for k, v in label_map.items()}
y_encoded = np.array([label_map[lab] for lab in y])

features = list(X_df.columns)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_df), columns=features)

# conservative augmentation
normal_mask = (np.array([lab == 'normal' for lab in y]) if 'normal' in label_map else np.array([False]*len(y)))
if normal_mask.any():
    rng = np.random.default_rng(RANDOM_STATE)
    noise_std = 0.02
    X_scaled.loc[normal_mask, features] += rng.normal(0, noise_std, size=(normal_mask.sum(), len(features)))
    mixes, mix_labels = [], []
    anomalous_idx = np.where(~normal_mask)[0]
    normal_idx = np.where(normal_mask)[0]
    n_mixes = min(400, len(normal_idx)*2, len(anomalous_idx)*2)
    for _ in range(n_mixes):
        a = X_scaled.iloc[rng.choice(normal_idx)].values
        b = X_scaled.iloc[rng.choice(anomalous_idx)].values
        alpha = rng.uniform(0.6, 0.85)
        mixes.append(alpha*a + (1-alpha)*b)
        mix_labels.append('normal')
    if mixes:
        mix_df = pd.DataFrame(mixes, columns=features)
        mix_y = np.array([label_map['normal']] * len(mixes))
        X_scaled = pd.concat([X_scaled.reset_index(drop=True), mix_df.reset_index(drop=True)], ignore_index=True)
        y_encoded = np.concatenate([y_encoded, mix_y], axis=0)
else:
    print("No 'normal' class -> skipping augmentation")

# ---------------- train/test split ----------------
X_train_all, X_test, y_train_all, y_test = train_test_split(
    X_scaled, y_encoded, test_size=TEST_SIZE, stratify=y_encoded, random_state=RANDOM_STATE
)
X_train_all, X_test = X_train_all.reset_index(drop=True), X_test.reset_index(drop=True)
y_train_all, y_test = y_train_all.reshape(-1), y_test.reshape(-1)

classes = np.unique(y_train_all)
weights = compute_class_weight('balanced', classes=classes, y=y_train_all)
class_weights = dict(zip(classes, weights))

n_classes = len(classes)
class_names = [reverse_label_map[i] for i in sorted(reverse_label_map)]

print(f" {class_names}")
print(f"{len(X_train_all)}, {len(X_test)}")

# ---------------- base + meta models ----------------
rf_base = RandomForestClassifier(n_estimators=100, max_depth=10, max_features='sqrt',
                                 random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1)
knn_base = KNeighborsClassifier(n_neighbors=15, weights='distance', p=2)
xgb_base = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                         subsample=0.8, colsample_bytree=0.8, gamma=0.1,
                         min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0,
                         random_state=RANDOM_STATE, use_label_encoder=False,
                         eval_metric='mlogloss', tree_method='hist', n_jobs=-1)
base_models = [('rf', rf_base), ('knn', knn_base), ('xgb', xgb_base)]
n_base = len(base_models)

meta_models = {
    'logistic': Pipeline([('scaler', StandardScaler()),
                          ('lr', LogisticRegression(max_iter=1000, C=0.1,
                                                    penalty='l2', solver='lbfgs',
                                                    multi_class='multinomial', random_state=RANDOM_STATE))]),
    'svm': Pipeline([('scaler', StandardScaler()),
                     ('svc', SVC(C=1.0, kernel='rbf', gamma='scale', class_weight='balanced',
                                 probability=True, random_state=RANDOM_STATE))]),
    'mlp_meta': Pipeline([('scaler', StandardScaler()),
                          ('mlp', MLPClassifier(hidden_layer_sizes=(20,10), activation='relu',
                                                solver='adam', alpha=0.01, batch_size=256,
                                                learning_rate='adaptive', learning_rate_init=0.001,
                                                max_iter=500, early_stopping=True, validation_fraction=0.2,
                                                n_iter_no_change=10, random_state=RANDOM_STATE))])
}

# ---------------- OOF stacking with resource-monitored training ----------------
def train_fold(X_tr, y_tr, X_val, y_val, fold_idx):
    fold_oof_meta = np.zeros((len(X_val), n_classes*n_base))
    fold_accs = {}
    for m_idx, (name, model) in enumerate(base_models):
        print(f" Fold {fold_idx} - Training base: {name}")
        print(f" Training Fold {fold_idx} / {fold_idx} ...")
        def train_base(cpu_usage_list=[], ram_usage_list=[]):
            m = clone(model)
            if name=='xgb':
                m.fit(X_tr, y_tr, sample_weight=np.array([class_weights[y] for y in y_tr]))
            else:
                m.fit(X_tr, y_tr)
            cpu_usage_list.append(psutil.cpu_percent())
            ram_usage_list.append(psutil.virtual_memory().percent)
            return m
        fitted_base_model = measure_training_resources(train_base)['result']
        proba_val = fitted_base_model.predict_proba(X_val)
        start_col = m_idx * n_classes
        fold_oof_meta[:, start_col:start_col+n_classes] = proba_val
    # train all meta models on this fold
    for key, meta in meta_models.items():
        print(f" Fold {fold_idx} - Training meta: {key}")
        tmp_meta = clone(meta)
        tmp_meta.fit(fold_oof_meta, y_val)  # training meta models on fold 
        val_pred = tmp_meta.predict(fold_oof_meta)
        acc = accuracy_score(y_val, val_pred)
        fold_accs[key] = acc
    return fold_oof_meta, fold_accs

kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
oof_meta = np.zeros((len(X_train_all), n_classes*n_base))
fold_scores = []

for fold_idx, (tr_idx, val_idx) in enumerate(kfold.split(X_train_all, y_train_all), start=1):
    X_tr, X_val = X_train_all.iloc[tr_idx], X_train_all.iloc[val_idx]
    y_tr, y_val = y_train_all[tr_idx], y_train_all[val_idx]
    fold_oof, fold_accs = train_fold(X_tr, y_tr, X_val, y_val, fold_idx)
    oof_meta[val_idx] = fold_oof
    fold_scores.append(fold_accs)

print("     All Folds Completed Successfully!")

# ---------------- fit base models on full training data ----------------
def train_full_base(name, model):
    def train_base(cpu_usage_list=[], ram_usage_list=[]):
        m = clone(model)
        if name=='xgb':
            m.fit(X_train_all, y_train_all, sample_weight=np.array([class_weights[y] for y in y_train_all]))
        else:
            m.fit(X_train_all, y_train_all)
        cpu_usage_list.append(psutil.cpu_percent())
        ram_usage_list.append(psutil.virtual_memory().percent)
        return m
    return measure_training_resources(train_base)['result']

fitted_base = {}
for name, model in base_models:
    print(f" {name}")
    fitted_base[name] = train_full_base(name, model)

meta_features_train = np.hstack([fitted_base[name].predict_proba(X_train_all) for name,_ in base_models])

# ---------------- train full meta models ----------------
def train_full_meta(key, meta):
    def train_meta(cpu_usage_list=[], ram_usage_list=[]):
        m = clone(meta)
        m.fit(meta_features_train, y_train_all)
        cpu_usage_list.append(psutil.cpu_percent())
        ram_usage_list.append(psutil.virtual_memory().percent)
        return m
    return measure_training_resources(train_meta)['result']

fitted_meta = {}
for key, meta in meta_models.items():
    print(f" {key}")
    fitted_meta[key] = train_full_meta(key, meta)

# ---------------- prediction ----------------
def do_prediction():
    base_probas = [fitted_base[name].predict_proba(X_test) for name,_ in base_models]
    meta_feats = np.hstack(base_probas)
    meta_probas = [fitted_meta[key].predict_proba(meta_feats) for key in fitted_meta]
    avg_meta_proba = np.mean(np.stack(meta_probas, axis=0), axis=0)
    preds = np.argmax(avg_meta_proba, axis=1)
    return {'y_pred': preds, 'y_proba': avg_meta_proba}

pred_summary = monitor_prediction(do_prediction)
pred_result = pred_summary['result']
y_test_pred = pred_result['y_pred']
y_test_proba = pred_result['y_proba']

# ---------------- final metrics ----------------
metrics = compute_all_metrics(y_true=y_test, y_pred=y_test_pred, y_proba=y_test_proba)
print("Validation Metrics ", metrics)

# ---------------- save outputs ----------------
artifact = {
    'base_models': fitted_base,
    'meta_models': fitted_meta,
    'scaler': scaler,
    'label_map': label_map,
    'reverse_label_map': reverse_label_map,
    'class_weights': class_weights,
    'features': features,
    'n_classes': n_classes,
    'class_names': class_names,
    'stacking_oof_meta': oof_meta
}
joblib.dump(artifact, ARTIFACT_OUT)

test_out = X_test.copy().reset_index(drop=True)
test_out['True_LabelEncoded'] = y_test
test_out['Pred_LabelEncoded'] = y_test_pred
test_out['Pred_Label'] = [reverse_label_map[i] for i in y_test_pred]
test_out.to_csv(PRED_OUT, index=False)

print("\nSaved Files:")
print(" - Hybrid_results.csv")
print(" - Hybrid_time.csv")
print(" - Hybrid_resources.csv")
print("\nProcess Finished.\n")


print("All artifacts, predictions, metrics saved.")
