from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    recall_score,
    f1_score,
    accuracy_score
)
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from glob import glob
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

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

# Load dataset
all_samples = []
for file in glob(os.path.join(truthful_path, "*.json")):
    all_samples.append(extract_features(file, 0))
for file in glob(os.path.join(deceitful_path, "*.json")):
    all_samples.append(extract_features(file, 1))

df = pd.DataFrame(all_samples)
df = shuffle(df, random_state=42).reset_index(drop=True)

X = df.drop("label", axis=1)
y = df["label"]

# 5-Fold Stratified Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_scores, acc_scores, recall_scores = [], [], []
fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
    print(f"\n🔁 Fold {fold}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_val_scaled)

    f1 = f1_score(y_val, y_pred, average='macro')
    acc = accuracy_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred, average=None)

    f1_scores.append(f1)
    acc_scores.append(acc)
    recall_scores.append(recall)

    fold_metrics.append({
        "Fold": fold,
        "Accuracy": acc,
        "Macro F1": f1,
        "Recall (truthful)": recall[0],
        "Recall (deceitful)": recall[1]
    })

    print(classification_report(y_val, y_pred, target_names=["Truthful", "Deceitful"]))

# Export fold results to CSV
fold_df = pd.DataFrame(fold_metrics)
fold_df.to_csv("classical_graph/rf_cv_fold_metrics.csv", index=False)

# Compute average results
recall_array = np.array(recall_scores)
metrics_df = pd.DataFrame({
    "Metric": ["Recall (truthful)", "Recall (deceitful)", "Accuracy", "Macro F1"],
    "Mean Score": [
        recall_array[:, 0].mean(),
        recall_array[:, 1].mean(),
        np.mean(acc_scores),
        np.mean(f1_scores)
    ]
})

print("\nCross-Validation Results (Averaged over 5 folds):")
print(metrics_df)

# Bar Chart
plt.figure(figsize=(10, 6))
sns.barplot(x="Metric", y="Mean Score", data=metrics_df, palette="Blues")
plt.ylim(0, 1)
plt.title("Random Forest Model Performance (5-Fold CV)")
plt.xlabel("Metric")
plt.ylabel("Score")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("classical_graph/rf_cv_metrics_chart.png")
plt.close()

