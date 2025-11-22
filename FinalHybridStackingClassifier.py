import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.base import clone
from collections import defaultdict
import joblib
import warnings
warnings.filterwarnings("ignore")

# ----------------- SETTINGS -----------------
RANDOM_STATE = 42
WINDOW_SIZE = 30     
STEP_SIZE = 1        
TEST_SIZE = 0.2
N_SPLITS = 5         

# ----------------- 1) Read & basic preprocess -----------------
print("Loading dataset...")
data = pd.read_csv("load_test_dataset.csv")

data = data[data["ResponseCode"] == 200].reset_index(drop=True)

def time_to_seconds(t):
    h, m, s = map(int, t.split(':'))
    return h * 3600 + m * 60 + s

data['TimeSeconds'] = data['TimeStamp'].apply(time_to_seconds)
data = data.sort_values(by='TimeSeconds').reset_index(drop=True)

# ----------------- 2) Create sliding-window features (overlapping: step=1) -----------------
print("Extracting sliding-window features (overlap: step=1)...")
def create_sliding_features(df, window_size=30, step=1):
    X_rows = []
    y_rows = []
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
    X_df = pd.DataFrame(X_rows)
    y_arr = np.array(y_rows)
    return X_df, y_arr

X_df, y = create_sliding_features(data, window_size=WINDOW_SIZE, step=STEP_SIZE)
print(f"Created {len(X_df)} sliding windows -> features shape: {X_df.shape}")

# ----------------- 3) Encode labels and map -----------------
unique_labels = sorted(np.unique(y))
label_map = {lab: idx for idx, lab in enumerate(unique_labels)}
reverse_label_map = {v: k for k, v in label_map.items()}
y_encoded = np.array([label_map[lab] for lab in y])

# ----------------- 4) Normalization (scale features) -----------------
features = list(X_df.columns)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_df), columns=features)

# ----------------- 5) Conservative augmentation on sliding windows -----------------
print("Applying conservative augmentation to sliding windows...")
normal_mask = (y == 'normal') if ('normal' in label_map) else np.array([False]*len(y))
if normal_mask.any():
    rng = np.random.default_rng(RANDOM_STATE)
    noise_std = 0.02
    X_scaled.loc[normal_mask, features] = X_scaled.loc[normal_mask, features] + rng.normal(0, noise_std, size=(normal_mask.sum(), len(features)))
    
    mixes = []
    mix_labels = []
    anomalous_idx = np.where(~normal_mask)[0]
    normal_idx = np.where(normal_mask)[0]
    n_mixes = min(400, len(normal_idx) * 2, len(anomalous_idx) * 2)  
    for _ in range(n_mixes):
        a = X_scaled.iloc[rng.choice(normal_idx)].values
        b = X_scaled.iloc[rng.choice(anomalous_idx)].values
        alpha = rng.uniform(0.6, 0.85)
        mixed = alpha * a + (1 - alpha) * b
        mixes.append(mixed)
        mix_labels.append('normal')
    if mixes:
        mix_df = pd.DataFrame(mixes, columns=features)
        mix_y = np.array([label_map['normal']] * len(mixes))
        X_scaled = pd.concat([X_scaled.reset_index(drop=True), mix_df.reset_index(drop=True)], ignore_index=True)
        y_encoded = np.concatenate([y_encoded, mix_y], axis=0)
else:
    print("No 'normal' class found for augmentation. Skipping augmentation step.")

# ----------------- 6) Train/Test split -----------------
print("Splitting into train/test sets...")
X_train_all, X_test, y_train_all, y_test = train_test_split(
    X_scaled, y_encoded, test_size=TEST_SIZE, stratify=y_encoded, random_state=RANDOM_STATE
)
X_train_all = X_train_all.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train_all = y_train_all.reshape(-1)
y_test = y_test.reshape(-1)

classes = np.unique(y_train_all)
weights = compute_class_weight('balanced', classes=classes, y=y_train_all)
class_weights = dict(zip(classes, weights))

n_classes = len(classes)
class_names = [reverse_label_map[i] for i in sorted(reverse_label_map)]

print(f"Classes: {class_names}")
print(f"Training samples: {len(X_train_all)}, Test samples: {len(X_test)}")

# ----------------- 7) Define base models (with requested hyperparameters) -----------------
print("Initializing base (level-0) models...")

rf_base = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    max_features='sqrt',
    random_state=RANDOM_STATE,
    class_weight='balanced',
    n_jobs=-1
)

knn_base = KNeighborsClassifier(
    n_neighbors=15,
    weights='distance',
    p=2,
    metric='minkowski',
    n_jobs=-1
)

xgb_base = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    scale_pos_weight=1,
    use_label_encoder=False,
    eval_metric='mlogloss',
    tree_method='hist',
    n_jobs=-1
)

base_models = [('rf', rf_base), ('knn', knn_base), ('xgb', xgb_base)]
n_base = len(base_models)

# ----------------- 8) Define meta models (level-1) -----------------
print("Initializing meta (level-1) models...")

meta_models = {
    'logistic': Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(
            max_iter=1000,
            C=0.1,
            penalty='l2',
            solver='lbfgs',
            multi_class='multinomial',
            random_state=RANDOM_STATE
        ))
    ]),
   
    'svm': Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(C=10, kernel='rbf', gamma=0.01, 
                    class_weight='balanced', probability=True, 
                    random_state=RANDOM_STATE))
])
    ]),
    'mlp_meta': Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=(20, 10),
            activation='relu',
            solver='adam',
            alpha=0.01,
            batch_size=256,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=10,
            random_state=RANDOM_STATE
        ))
    ])
}

# ----------------- 9) K-fold stacking to create out-of-fold meta-features -----------------
print(f"Starting stacking with {N_SPLITS}-fold StratifiedKFold to build OOF meta-features...")
kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

oof_meta = np.zeros((len(X_train_all), n_classes * n_base), dtype=float)
fold_scores = []

for fold_idx, (tr_idx, val_idx) in enumerate(kfold.split(X_train_all, y_train_all), start=1):
    print(f"\n--- Fold {fold_idx}/{N_SPLITS} ---")
    X_tr, X_val = X_train_all.iloc[tr_idx], X_train_all.iloc[val_idx]
    y_tr, y_val = y_train_all[tr_idx], y_train_all[val_idx]

    sample_weights = np.array([class_weights[y] for y in y_tr])

    for m_idx, (name, model) in enumerate(base_models):
        print(f"Training base model: {name}")
        m = clone(model)
        if name == 'xgb':
            m.fit(X_tr, y_tr, sample_weight=sample_weights)
        else:
            m.fit(X_tr, y_tr)
        proba_val = m.predict_proba(X_val)  
        start_col = m_idx * n_classes
        end_col = start_col + n_classes
        oof_meta[val_idx, start_col:end_col] = proba_val

    tmp_meta = clone(meta_models['logistic'])
    tmp_meta.fit(oof_meta[tr_idx], y_tr)
    val_pred = tmp_meta.predict(oof_meta[val_idx])
    acc = accuracy_score(y_val, val_pred)
    print(f"Fold {fold_idx} logistic-on-oof accuracy: {acc:.4f}")
    fold_scores.append(acc)

print(f"\nAverage fold logistic-on-oof acc: {np.mean(fold_scores):.4f}")

# ----------------- 10) Train base models on full training data -----------------
print("\nTraining base models on full training set...")
fitted_base = {}
for name, model in base_models:
    print(f"Fitting {name} on full training data...")
    m = clone(model)
    if name == 'xgb':
        m.fit(X_train_all, y_train_all, sample_weight=np.array([class_weights[y] for y in y_train_all]))
    else:
        m.fit(X_train_all, y_train_all)
    fitted_base[name] = m

meta_features_train = np.hstack([fitted_base[name].predict_proba(X_train_all) for name, _ in base_models])

# ----------------- 11) Train meta models on meta_features_train -----------------
print("\nTraining meta models on stacked meta-features...")
fitted_meta = {}
for key, meta in meta_models.items():
    print(f" Training meta model: {key}")
    m = clone(meta)
    m.fit(meta_features_train, y_train_all)
    fitted_meta[key] = m

# ----------------- 12) Final soft voting across meta-models -----------------
def predict_soft_vote_meta(fitted_base, fitted_meta, X):
    base_probas = [fitted_base[name].predict_proba(X) for name, _ in base_models]
    meta_feats = np.hstack(base_probas) 
    meta_probas = []
    for key in fitted_meta:
        probs = fitted_meta[key].predict_proba(meta_feats)
        meta_probas.append(probs)
    avg_meta_proba = np.mean(np.stack(meta_probas, axis=0), axis=0)  
    preds = np.argmax(avg_meta_proba, axis=1)
    return preds, avg_meta_proba

print("\nEvaluating stacked + meta-level soft voting pipeline...")
y_train_pred, y_train_proba = predict_soft_vote_meta(fitted_base, fitted_meta, X_train_all)
y_test_pred, y_test_proba = predict_soft_vote_meta(fitted_base, fitted_meta, X_test)
n_classes = len(class_names)
# ----------------- 13) Save artifacts & predictions -----------------
print("\nSaving models and outputs...")
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
joblib.dump(artifact, 'stacked_softvote_artifact.pkl')

test_out = X_test.copy().reset_index(drop=True)
test_out['True_LabelEncoded'] = y_test
test_out['Pred_LabelEncoded'] = y_test_pred
test_out['Pred_Label'] = [reverse_label_map[i] for i in y_test_pred]
test_out.to_csv("stacked_softvote_test_predictions.csv", index=False)

print("All artifacts saved. Done.")

