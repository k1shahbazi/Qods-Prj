import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
)

def compute_all_metrics(y_true, y_pred, y_proba, average="macro"):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0)
    }

    # Compute AUC (multi-class support)
    try:
        n_classes = y_proba.shape[1]
        auc = roc_auc_score(
            np.eye(n_classes)[y_true],
            y_proba,
            average="macro",
            multi_class="ovr"
        )
        metrics["auc"] = auc
    except Exception:
        metrics["auc"] = None

    return metrics
