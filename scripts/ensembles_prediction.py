"""
Script 3 : Construction des ensembles de prédiction
Entrées : test.csv + ai_scores_all.csv + seuil.csv
Sortie : resultats.csv
"""

import pandas as pd

# 1. Charger les données
print("Chargement...")
test = pd.read_csv("fichiers_utilises/test.csv")
scores = pd.read_csv("fichiers_utilises/ai_scores_all.csv")
seuil_df = pd.read_csv("seuil.csv")
seuil_score = seuil_df['seuil_score'].iloc[0]

print(f"Test : {len(test)} observations")
print(f"Seuil : garder scores >= {seuil_score:.4f}")

# 2. Fusionner
test_scores = scores[scores['observation_id'].isin(test['observation_id'])]
merged = test_scores.merge(test, on='observation_id')

# 3. Construire les ensembles
resultats = []

for obs_id in merged['observation_id'].unique():
    lignes = merged[merged['observation_id'] == obs_id]
    gardees = lignes[lignes['score'] >= seuil_score]['spicies_id'].tolist()
    
    if len(gardees) == 0:
        meilleure = lignes.loc[lignes['score'].idxmax(), 'spicies_id']
        gardees = [meilleure]
    
    vrai = lignes['ground_truth_val'].iloc[0]
    
    resultats.append({
        'observation_id': obs_id,
        'vraie_espece': vrai,
        'ensemble': gardees,
        'taille': len(gardees),
        'couvert': vrai in gardees
    })

# 4. Sauvegarder
res_df = pd.DataFrame(resultats)
res_df.to_csv("resultats.csv", index=False)

print(f"\n✅ {len(resultats)} observations traitées")
print(f"Couverture : {res_df['couvert'].mean():.3f}")
print(f"Taille moyenne : {res_df['taille'].mean():.2f}")