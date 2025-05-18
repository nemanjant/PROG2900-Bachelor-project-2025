import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from glob import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score
)

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Paths
truthful_path = "data/truthful"
deceitful_path = "data/deceitful"

# Feature extraction function
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

# Load and extract all data
all_samples = []
for file in glob(os.path.join(truthful_path, "*.json")):
    all_samples.append(extract_features(file, 0))
for file in glob(os.path.join(deceitful_path, "*.json")):
    all_samples.append(extract_features(file, 1))

df = pd.DataFrame(all_samples)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

X = df.drop("label", axis=1)
y = df["label"]

# Cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_scores, acc_scores, recall_scores, auc_scores = [], [], [], []
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
    y_proba = clf.predict_proba(X_val_scaled)[:, 1]

    f1 = f1_score(y_val, y_pred, average='macro')
    acc = accuracy_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred, average=None)
    auc = roc_auc_score(y_val, y_proba)

    f1_scores.append(f1)
    acc_scores.append(acc)
    recall_scores.append(recall)
    auc_scores.append(auc)

    fold_metrics.append({
        "Fold": fold,
        "Accuracy": acc,
        "Macro F1": f1,
        "Recall (truthful)": recall[0],
        "Recall (deceitful)": recall[1],
        "AUC": auc
    })

    print(classification_report(y_val, y_pred, target_names=["Truthful", "Deceitful"]))

# Save fold results
fold_df = pd.DataFrame(fold_metrics)
fold_df.to_csv("classical_graph/rf_cv_fold_metrics.csv", index=False)

# Summary
recall_array = np.array(recall_scores)
metrics_df = pd.DataFrame({
    "Metric": ["Recall (truthful)", "Recall (deceitful)", "Accuracy", "Macro F1", "AUC"],
    "Mean Score": [
        recall_array[:, 0].mean(),
        recall_array[:, 1].mean(),
        np.mean(acc_scores),
        np.mean(f1_scores),
        np.mean(auc_scores)
    ]
})

print("\nCross-Validation Results (Averaged over 5 folds):")
print(metrics_df)

# Plot summary
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



