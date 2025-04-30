import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from glob import glob
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    classification_report,
    recall_score,
    f1_score,
    accuracy_score
)

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Paths to data
truthful_path = "data/truthful"
deceitful_path = "data/deceitful"

# Feature extraction
def extract_features(filepath, label):
    with open(filepath, "r") as f:
        data = json.load(f)
    return {
        "totalTime": data.get("totalTime", 0),
        "averageSpeed": data.get("averageSpeed", 0),
        "hesitation": data.get("hesitation", 0),
        "jerkSpikeCount": data.get("jerkSpikeCount", 0),
        "acceleration_mean": np.mean(data.get("accelerations", [0])),
        "acceleration_std": np.std(data.get("accelerations", [0])),
        "jerk_mean": np.mean(data.get("jerks", [0])),
        "jerk_std": np.std(data.get("jerks", [0])),
        "curvature_mean": np.mean(data.get("curvatures", [0])),
        "curvature_std": np.std(data.get("curvatures", [0])),
        "label": label
    }

# Load and prepare dataset
all_samples = []
for file in glob(os.path.join(truthful_path, "*.json")):
    all_samples.append(extract_features(file, 0))
for file in glob(os.path.join(deceitful_path, "*.json")):
    all_samples.append(extract_features(file, 1))

df = pd.DataFrame(all_samples)
df = shuffle(df, random_state=42).reset_index(drop=True)

# Split dataset
X = df.drop("label", axis=1)
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest (deterministic)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)
y_prob = clf.predict_proba(X_test_scaled)[:, 1]

# --- Evaluation & Visualization ---

# 1. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="orange")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("roc.png")
plt.close()

# 2. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Truthful", "Deceitful"], yticklabels=["Truthful", "Deceitful"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

# 3. Feature Importance
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
top_features = X.columns[indices][:10]
plt.figure()
sns.barplot(x=importances[indices][:10], y=top_features)
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.savefig("feature_importance.png")
plt.close()

# 4. Feature Correlation (Triangle)
top_corr_features = df.corr()["label"].abs().sort_values(ascending=False)[1:11].index
corr = df[top_corr_features].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
plt.figure(figsize=(10, 8))
sns.heatmap(corr, mask=mask, annot=True, cmap="coolwarm", square=True, linewidths=0.5, cbar_kws={"shrink": 0.75})
plt.title("Top 10 Feature Correlation Matrix")
plt.savefig("feature_correlation_matrix.png")
plt.close()

# 5. Classification Report
print("\n✅ Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Truthful", "Deceitful"]))

# 6. Metrics Chart (Accuracy, Recall, Macro F1) with horizontal grid
accuracy = accuracy_score(y_test, y_pred)
recall_class = recall_score(y_test, y_pred, average=None)
f1_macro = f1_score(y_test, y_pred, average='macro')

metrics_df = pd.DataFrame({
    "Metric": ["Recall (truthful)", "Recall (deceitful)", "Accuracy", "Macro F1"],
    "Score": [recall_class[0], recall_class[1], accuracy, f1_macro]
})

plt.figure(figsize=(10, 6))
sns.barplot(x="Metric", y="Score", data=metrics_df, palette="Blues")
plt.ylim(0, 1)
plt.title("Model Performance Metrics")
plt.xlabel("Metric")
plt.ylabel("Score")
plt.xticks(rotation=0, ha='center')
plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
plt.tight_layout()
plt.savefig("f1_score_chart.png")
plt.close()

print("\n📊 Plots saved:")
print(" - roc.png")
print(" - confusion_matrix.png")
print(" - feature_importance.png")
print(" - feature_correlation_matrix.png")
print(" - f1_score_chart.png")


