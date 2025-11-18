# ----------------- 15) Per-Class Metrics (Precision, Recall, F1, Accuracy, AUC) -----------------
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score, confusion_matrix

print("\nCalculating per-class metrics for TEST set...")

y_test_binarized = label_binarize(y_test, classes=np.arange(n_classes))

per_class_results = []

for i, class_id in enumerate(np.arange(n_classes)):
    class_name = reverse_label_map[class_id]

    y_true_bin = (y_test == class_id).astype(int)
    y_pred_bin = (y_test_pred == class_id).astype(int)

    precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    acc = accuracy_score(y_true_bin, y_pred_bin)

    try:
        auc = roc_auc_score(y_test_binarized[:, i], y_test_proba[:, i])
    except ValueError:
        auc = np.nan  

    per_class_results.append({
        'Class': class_name,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1,
        'Accuracy': acc,
    })
per_class_df = pd.DataFrame(per_class_results)
per_class_df.to_csv("per_class_metrics.csv", index=False)
print("\n=== Per-Class Metrics ===")س