from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ----------------- Confusion Matrix -----------------
print("\nPlotting Confusion Matrix...")

cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(8,6))
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix - Test Data", fontsize=14)
plt.tight_layout()
plt.show()

