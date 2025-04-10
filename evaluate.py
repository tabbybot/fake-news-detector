# auto creation of plots folder using os module
import os
folder_name = "plots"
folder_path = os.path.join(os.getcwd(), folder_name)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# other libraries
import joblib
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

model = joblib.load("model.pkl")

bundle = joblib.load("test_data_bundle.pkl")
X_test_vec = bundle["X_test_vec"]
X_test_raw = bundle["X_test_raw"]
y_test = bundle["y_test"]

y_pred = model.predict(X_test_vec)

# Classification Report is here
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Bar graph of Performance Metrics is here
scores = {'Accuracy':accuracy, 'Precision':precision,
          'Recall':recall, 'F1':f1}
plt.bar(scores.keys(), scores.values(), color='skyblue')
plt.title("Model Performance Metrics")
plt.ylim(0, 1)
plt.savefig("plots/metrics.png")
plt.show()

# Confusion matrix is here
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['Real', 'Fake'])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("plots/confusion.png")
plt.show()

# ROC_AUC curve is here
fpr, tpr, _ = roc_curve(y_test,
                        model.predict_proba(X_test_vec)[:,1])
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend()
plt.savefig("plots/roc_auc.png")
plt.show()

# Misclassified examples info is here
df = pd.DataFrame({"text": X_test_raw,
                   "actual": y_test,
                   "predicted": y_pred})
misclassified = df[df["actual"] != df["predicted"]]
print(misclassified.sample(5))

# Prediction Confidence Distribution is here
probs = model.predict_proba(X_test_vec)[:,1]
sns.histplot(probs, kde=True, bins=20)
plt.title("Prediction Confidence Distribution")
plt.xlabel("Confidence score (for 'Fake')")
plt.savefig("plots/confidence.png")
plt.show()

import numpy as np

print("High confidence predictions:", np.sum((probs > 0.99) | (probs < 0.01)))
print("Medium confidence predictions:", np.sum((probs >= 0.4) & (probs <= 0.6)))
