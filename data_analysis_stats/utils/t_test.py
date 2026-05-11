import numpy as np
from scipy.stats import ttest_rel

# Macro F1-scores across 5 folds for two models
rf_scores = [0.614, 0.579, 0.517, 0.571, 0.599]    # Random Forest
dl_scores = [0.578, 0.492, 0.514, 0.621, 0.520]    # Deep Learning

# Perform paired t-test
t_stat, p_value = ttest_rel(rf_scores, dl_scores)

# Print results
print("Paired t-test for macro F1-scores:")
print(f"Random Forest scores:     {rf_scores}")
print(f"Deep Learning scores:     {dl_scores}")
print(f"T-statistic: {t_stat:.3f}")
print(f"P-value:     {p_value:.3f}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("→ Statistically significant difference (reject null hypothesis)")
else:
    print("→ No statistically significant difference (fail to reject null hypothesis)")
