import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Charger ground truth
df_gt = pd.read_csv("observations_experts.csv")

# 2. Charger les scores
df_scores = pd.read_csv("ai_scores_all.csv")

# 3. Fusionner
df_merged = df_scores.merge(df_gt, on='observation_id')

# 4. Garder uniquement le score de la VRAIE espèce pour chaque observation
true_scores = df_merged[df_merged['spicies_id'] == df_merged['ground_truth_val']].copy()

print(f"Nombre d'observations valides : {len(true_scores)}")

# 5. Séparer calibration (50%) et test (50%)
calib_ids, test_ids = train_test_split(
    true_scores['observation_id'].unique(), 
    test_size=0.5, 
    random_state=42
)

# 6. Calibration
calib_true_scores = true_scores[true_scores['observation_id'].isin(calib_ids)]['score'].values

# 7. Calcul du seuil avec α = 0.05 (95% de couverture)
alpha = 0.05
n_calib = len(calib_true_scores)
quantile_level = (1 - alpha) * (1 + 1/n_calib)
threshold = np.quantile(calib_true_scores, quantile_level, method='higher')

print(f"\n=== Résultats ===")
print(f"Nombre d'observations en calibration : {n_calib}")
print(f"α = {alpha} (couverture cible : {1-alpha*100}%)")
print(f"Niveau du quantile : {quantile_level:.6f}")
print(f"Seuil : {threshold:.6f}")
print(f"\nOn garde les espèces avec score >= {threshold:.6f}")