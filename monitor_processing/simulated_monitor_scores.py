#Creates simulated monitor scores
# %%
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any
project_dir = Path(__file__).parent.parent
# %%
np.random.seed(123)
og_honest_scores = np.random.beta(7.39, 6.86,500)
og_attack_scores = np.random.beta(8.91, 2.82, 500)
noised_honest_scores = np.random.normal(og_honest_scores, 0.06)
noised_attack_scores = np.random.normal(og_attack_scores, 0.06)
weaker_monitor_attack = np.random.beta(7.51, 4.22, 500)
weaker_monitor_honest = np.random.beta(7.39, 6.86,500)

df = pd.DataFrame({'honest_scores': og_honest_scores, 'attack_scores': og_attack_scores,
                   'noised_honest_scores': noised_honest_scores, 'noised_attack_scores': noised_attack_scores,
                   'weaker_monitor_attack': weaker_monitor_attack, 'weaker_monitor_honest': weaker_monitor_honest})
df.to_csv(project_dir / 'data' / 'processed_monitor_scores'/'beta_sim_scores.csv', index=False)

# %%
import sys
sys.path.insert(0, str(project_dir))
import single_monitor as sm


# %%

best_obj, a_m_star, p_m_star, a_s_star = sm.solve_1mon_optimal_policy(c_a = 10000, c_m = 1, Y = 301, attack_scores = og_attack_scores, honest_scores = og_honest_scores)

naive_a_m = sm.get_naive_single_monitor_policy(honest_scores = og_honest_scores, Y = 301, c_a = 10000, c_m = 1)
print(f"Optimized best_obj: {best_obj:.5f}, Naive best_obj: {sm.evaluate_1mon_ood(attack_scores_ood = og_attack_scores, a_m = naive_a_m, p_m = 1, a_s = 0):.5f}")


#%%
in_sample_alpha_benign = og_honest_scores
in_sample_alpha_attack = og_attack_scores
in_sample_beta_benign = weaker_monitor_honest
in_sample_beta_attack = weaker_monitor_attack

np.random.seed(1)
# Resample benign data with replacement
resampled_alpha_benign = np.random.choice(in_sample_alpha_benign, size=len(in_sample_alpha_benign), replace=True)
resampled_alpha_benign += np.random.normal(0, 1e-8, size=len(resampled_alpha_benign))
resampled_beta_benign = np.random.choice(in_sample_beta_benign, size=len(in_sample_beta_benign), replace=True)
resampled_beta_benign += np.random.normal(0, 1e-8, size=len(resampled_beta_benign))


# Resample benign data with replacement
resampled_alpha_benign = np.random.choice(in_sample_alpha_benign, size=len(in_sample_alpha_benign), replace=True)
resampled_alpha_benign += np.random.normal(0, 1e-8, size=len(resampled_alpha_benign))
resampled_beta_benign = np.random.choice(in_sample_beta_benign, size=len(in_sample_beta_benign), replace=True)
resampled_beta_benign += np.random.normal(0, 1e-8, size=len(resampled_beta_benign))

# Resample attack data and split into train and test sets
# First, resample all attack data
total_attack_samples = len(in_sample_alpha_attack)        
# Randomly select indices for test set
test_indices = np.random.choice(total_attack_samples, size=100, replace=False)
train_indices = np.array([i for i in range(total_attack_samples) if i not in test_indices])

# Split into train and test sets
train_alpha_attack = in_sample_alpha_attack[train_indices]
test_alpha_attack = in_sample_alpha_attack[test_indices]
train_beta_attack = in_sample_beta_attack[train_indices]
test_beta_attack = in_sample_beta_attack[test_indices]

#resample each set
train_alpha_attack = np.random.choice(train_alpha_attack, size=len(train_alpha_attack), replace=True)
train_alpha_attack += np.random.normal(0, 1e-8, size=len(train_alpha_attack))
train_beta_attack = np.random.choice(train_beta_attack, size=len(train_beta_attack), replace=True)
train_beta_attack += np.random.normal(0, 1e-8, size=len(train_beta_attack))
test_alpha_attack = np.random.choice(test_alpha_attack, size=len(test_alpha_attack), replace=True)
test_alpha_attack += np.random.normal(0, 1e-8, size=len(test_alpha_attack))
test_beta_attack = np.random.choice(test_beta_attack, size=len(test_beta_attack), replace=True)
test_beta_attack += np.random.normal(0, 1e-8, size=len(test_beta_attack))

# %%

best_obj, a_m_star, p_m_star, a_s_star = sm.solve_1mon_optimal_policy(c_a = 10000, c_m = 1, Y = 221, attack_scores = train_alpha_attack, honest_scores = resampled_alpha_benign)

naive_a_m = sm.get_naive_single_monitor_policy(honest_scores = resampled_alpha_benign, Y = 221, c_a = 10000, c_m = 1)
print(f"Optimized best_obj: {best_obj:.5f}, Naive best_obj: {sm.evaluate_1mon_ood(attack_scores_ood = train_alpha_attack, a_m = naive_a_m, p_m = 1, a_s = 0):.5f}")


pd.DataFrame({'honest_scores': resampled_alpha_benign}).to_csv(project_dir / 'single_monitor' / 'scores_for_debugging_honest.csv', index=False)
pd.DataFrame({'attack_scores': train_alpha_attack}).to_csv(project_dir / 'single_monitor' / 'scores_for_debugging_attack.csv', index=False)




# %%

