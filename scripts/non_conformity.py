"""
Script 1 : Calcul des scores de non-conformité
Entrées : calibration.csv + ai_scores_all.csv
Sortie : non_conformity.csv
"""

import pandas as pd

# 1. Charger les données
print("Chargement des fichiers...")
calib = pd.read_csv("fichiers_utilises/calibration.csv")
scores = pd.read_csv("fichiers_utilises/ai_scores_all.csv")

print(f"Calibration : {len(calib)} observations")
print(f"Scores : {len(scores):,} lignes")

# 2. Filtrer les scores pour garder uniquement les photos de calibration
calib_scores = scores[scores['observation_id'].isin(calib['observation_id'])]

# 3. Fusionner pour avoir le ground_truth à côté
merged = calib_scores.merge(calib, on='observation_id')

# 4. Garder uniquement la ligne où l'espèce = vraie espèce
vraie = merged[merged['spicies_id'] == merged['ground_truth_val']].copy()

# 5. Calculer la non-conformité
vraie['non_conformity'] = 1 - vraie['score']

# 6. Garder seulement les colonnes utiles
resultat = vraie[['observation_id', 'non_conformity']]

# 7. Sauvegarder
resultat.to_csv("non_conformity.csv", index=False)

print(f"\nTerminé ! {len(resultat)} observations")
print(f"Non-conformité - min: {resultat['non_conformity'].min():.4f}")
print(f"Non-conformité - max: {resultat['non_conformity'].max():.4f}")
print(f"Non-conformité - moyenne: {resultat['non_conformity'].mean():.4f}")
print("\nFichier sauvegardé : non_conformity.csv")