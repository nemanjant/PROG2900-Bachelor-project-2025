import numpy as np
from scipy import stats

def compute_confidence_interval(scores, confidence=0.95):
    scores = np.array(scores)
    n = len(scores)
    mean = np.mean(scores)
    std = np.std(scores, ddof=1)  # Sample standard deviation
    t_score = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    margin_error = t_score * (std / np.sqrt(n))
    ci_lower = mean - margin_error
    ci_upper = mean + margin_error
    
    print(f"Values: {scores}")
    print(f"Mean: {mean:.3f}")
    print(f"Standard Deviation: {std:.3f}")
    print(f"{int(confidence * 100)}% Confidence Interval: ±{margin_error:.3f}")
    print(f"Range: ({ci_lower:.3f}, {ci_upper:.3f})")
    print("-" * 40)


# Deep Learning confidence interval compuutation
f1_scores = [0.578, 0.492, 0.514, 0.621, 0.520]
acc_scores = [0.579, 0.486, 0.514, 0.621, 0.521]
recall_truthful  = [0.54, 0.57, 0.50, 0.63, 0.48]
recall_deceitful = [0.70, 0.42, 0.53, 0.61, 0.57]
auc_scores = [0.61, 0.52, 0.54, 0.65, 0.55]

# Random Forest confidence interval compuutation
# f1_scores = [0.614, 0.579, 0.517, 0.571, 0.599]
# acc_scores = [0.614, 0.579, 0.521, 0.571, 0.600]
# recall_truthful  = [0.643, 0.586, 0.614, 0.571, 0.543]
# recall_deceitful = [0.586, 0.571, 0.429, 0.571, 0.657]
# auc_scores = [0.64, 0.60, 0.57, 0.63, 0.65]


print("Macro F1:")
compute_confidence_interval(f1_scores)

print("Accuracy:")
compute_confidence_interval(acc_scores)

print("Recall (truthful) ")
compute_confidence_interval(recall_truthful)

print("Recall (deceitful):")
compute_confidence_interval(recall_deceitful)

print("AUC:")
compute_confidence_interval(auc_scores)

