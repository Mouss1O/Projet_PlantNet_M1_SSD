"""
Script 2 : Calcul du seuil (quantile)
Entrée : non_conformity.csv
Sortie : seuil.csv
"""

import pandas as pd
import numpy as np

# 1. Charger les scores de non-conformité
print("Chargement des scores...")
df = pd.read_csv("non_conformity.csv")

# 2. Paramètres
alpha = 0.05  # Risque d'erreur (couverture cible = 95%)
n = len(df)

print(f"α = {alpha} (couverture cible = {1-alpha:.0%})")
print(f"Nombre d'observations : n = {n}")

# 3. Calcul du quantile (formule corrigée)
quantile_level = (1 - alpha) * (1 + 1/n)
print(f"Niveau du quantile : {quantile_level:.6f}")

# 4. Calcul du seuil q_hat
scores = df['non_conformity'].values
q_hat = np.quantile(scores, quantile_level, method='higher')

# 5. Seuil équivalent sur les scores
seuil_score = 1 - q_hat

# 6. Affichage des résultats
print("\n" + "="*50)
print("RÉSULTATS")
print("="*50)
print(f"q_hat (seuil sur non-conformité) = {q_hat:.6f}")
print(f"Seuil sur les scores = {seuil_score:.6f}")
print(f"→ On garde les espèces avec score >= {seuil_score:.6f}")

# 7. Sauvegarde
resultats = pd.DataFrame({
    'alpha': [alpha],
    'n_calibration': [n],
    'quantile_level': [quantile_level],
    'q_hat': [q_hat],
    'seuil_score': [seuil_score]
})
resultats.to_csv("seuil.csv", index=False)

print("\n✅ Fichier sauvegardé : seuil.csv")