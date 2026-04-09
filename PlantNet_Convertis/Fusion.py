import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit

# 1. Charger les fichiers
experts = pd.read_csv("observations_experts.csv")
scores = pd.read_csv("ai_scores_all.csv")

print(f"Experts: {experts.shape}")
print(f"Scores: {scores.shape}")

# 2. Découpage ShuffleSplit sur les experts
rs = ShuffleSplit(n_splits=1, test_size=0.5, random_state=42)

for train_idx, test_idx in rs.split(experts):
    calib_experts = experts.iloc[train_idx]
    test_experts = experts.iloc[test_idx]

print(f"\nCalibration: {len(calib_experts)} observations")
print(f"Test: {len(test_experts)} observations")

# 3. Fusionner avec les scores (sans pivot)
# On garde le format long
calib_scores = calib_experts.merge(scores, on='observation_id')
test_scores = test_experts.merge(scores, on='observation_id')

print(f"\nCalibration avec scores: {calib_scores.shape}")
print(f"Test avec scores: {test_scores.shape}")

# 4. Sauvegarder (toujours au format long)
calib_scores.to_csv("calibration_with_scores_long.csv", index=False)
test_scores.to_csv("test_with_scores_long.csv", index=False)

print("\n✅ Fichiers sauvegardés (format long) :")
print("   - calibration_with_scores_long.csv")
print("   - test_with_scores_long.csv")