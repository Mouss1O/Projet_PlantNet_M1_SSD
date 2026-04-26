import pandas as pd

# Charger les données
calib = pd.read_csv("data/split_random/calibration.csv")
scores = pd.read_csv("ai_scores_all.csv")

# Fusionner
calib_scores = scores[scores['observation_id'].isin(calib['observation_id'])]
merged = calib_scores.merge(calib, on='observation_id')

# Garder la ligne où l'espèce = vraie espèce
vraie = merged[merged['spicies_id'] == merged['ground_truth_val']]

# Calculer S = 1 - score
vraie['non_conformity'] = 1 - vraie['score']

# Sauvegarder
vraie[['observation_id', 'non_conformity']].to_csv("non_conformity.csv", index=False)

print("Fini. Fichier : non_conformity.csv")