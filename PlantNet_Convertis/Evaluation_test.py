import pandas as pd
import numpy as np

# 1. Charger le seuil (ou le définir ici)
seuil = 0.96628

# 2. Charger le fichier test (format long)
test_scores = pd.read_csv("test_with_scores_long.csv")

print(f"Fichier test chargé : {test_scores.shape}")
print(test_scores.head())

# 3. Pour chaque observation, garder les espèces avec score >= seuil
# On regroupe par observation_id
prediction_sets = test_scores.groupby('observation_id').apply(
    lambda x: set(x[x['score'] >= seuil]['spicies_id'].tolist())
).reset_index(name='pred_set')

# 4. Ajouter le vrai label
true_labels = test_scores[['observation_id', 'ground_truth_val']].drop_duplicates()
prediction_sets = prediction_sets.merge(true_labels, on='observation_id')

# 5. Vérifier si un ensemble est vide
prediction_sets['is_empty'] = prediction_sets['pred_set'].apply(lambda x: len(x) == 0)

# 6. Calculer la couverture
prediction_sets['covered'] = prediction_sets.apply(
    lambda x: x['ground_truth_val'] in x['pred_set'], axis=1
)

coverage = prediction_sets['covered'].mean()
empty_rate = prediction_sets['is_empty'].mean()

# 7. Taille moyenne des ensembles (non vides)
set_sizes = prediction_sets['pred_set'].apply(len)
avg_size = set_sizes.mean()

print(f"\n=== Résultats sur le test ===")
print(f"Nombre d'observations test : {len(prediction_sets)}")
print(f"Seuil utilisé : {seuil}")
print(f"Couverture observée : {coverage:.3f} (attendue >= 0.95)")
print(f"Taille moyenne des ensembles : {avg_size:.2f}")
print(f"Taux d'ensembles vides : {empty_rate:.3f}")

# 8. Distribution des tailles
print(f"\nDistribution des tailles :")
print(prediction_sets['pred_set'].apply(len).value_counts().sort_index())