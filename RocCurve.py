from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

# -------------------- آماده‌سازی داده‌ها --------------------
y_test_bin = label_binarize(y_test, classes=range(n_classes))
y_train_bin = label_binarize(y_train_all, classes=range(n_classes))

# -------------------- رسم ROC برای هر کلاس --------------------
for i, cls_name in enumerate(class_names):
    plt.figure(figsize=(8, 6))

    fpr_test, tpr_test, _ = roc_curve(y_test_bin[:, i], y_test_proba[:, i])
    auc_test = auc(fpr_test, tpr_test)
    plt.plot(fpr_test, tpr_test, label=f"Test (AUC={auc_test:.2f})", color='red', linestyle='--', linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label="Random")

    plt.title(f"ROC Curve – Class: {cls_name}", fontsize=14, fontweight='bold')
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.show()
