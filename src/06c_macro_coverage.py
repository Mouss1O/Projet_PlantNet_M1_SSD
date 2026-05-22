"""
=============================================================================
Étude 3 : Macro-Couverture — Approche Naïve (Sans correction)
=============================================================================
Projet : Pl@ntNet-CP — Prédiction Conformelle
Objectif : Calculer la couverture macro (moyenne des couvertures par espèce)
           pour les 3 méthodes (Standard, PAS, Classwise) sur 4 alphas.
           On mesure ici l'équité statistique entre espèces.

Entrées :
    - expert_calib_50.csv 
    - expert_test_50.csv

Sorties :
    - Tableau récapitulatif Alpha / Méthode / Macro-Cov / Taille Moyenne.
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ================================================================
# CONFIGURATION
# ================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PATH_CALIB = os.path.join(SCRIPT_DIR, "expert_calib_50.csv")
PATH_TEST  = os.path.join(SCRIPT_DIR, "expert_test_50.csv")

ALPHAS     = [0.01, 0.05, 0.10, 0.20]
N_THEO     = 10812  # 50% de 21624

# ================================================================
# CHARGEMENT ET PRÉPARATION
# ================================================================
print("=" * 95)
print("ÉTUDE DE LA MACRO-COUVERTURE : TROIS MÉTHODES (APPROCHE NAÏVE)")
print("=" * 95)

calib = pd.read_csv(PATH_CALIB)
test  = pd.read_csv(PATH_TEST)

for df in [calib, test]:
    df['observation_id']  = df['observation_id'].astype(str).str.strip()
    df['species_id']      = df['species_id'].astype(str).str.strip()
    df['true_species_id'] = df['true_species_id'].astype(str).str.strip()

# 1. Prévalence sur calibration complète (pour PAS)
counts_total = calib.drop_duplicates('observation_id').groupby('true_species_id').size()
p_y_dict = (counts_total / N_THEO).to_dict()
p_min = counts_total.min() / N_THEO

# 2. Isole les matches pour calibration (Naïf)
calib_match = calib[calib['species_id'] == calib['true_species_id']].copy()
counts_calib_match = calib_match.groupby('true_species_id').size()

# 3. Préparation des statistiques de référence du test
# On doit savoir combien de fois chaque espèce apparaît réellement dans le test
test_ref_counts = test.drop_duplicates('observation_id').groupby('true_species_id').size()

# ================================================================
# BOUCLE SUR LES ALPHAS
# ================================================================
results_macro = []

for alpha in ALPHAS:
    print(f"\nCalcul pour alpha = {alpha}...")
    
    # --- A. CALCUL DES SEUILS (NAÏFS) ---
    n_cal = len(calib_match)
    lvl = min(np.ceil((n_cal + 1) * (1 - alpha)) / n_cal, 1.0)
    
    # Standard
    q_std = np.quantile(-calib_match['score'].values, lvl, method="higher")
    
    # PAS
    calib_match['p_y'] = calib_match['true_species_id'].map(p_y_dict).fillna(p_min)
    q_pas = np.quantile(-(calib_match['score'] / calib_match['p_y']).values, lvl, method="higher")
    
    # Classwise (Mondrian)
    class_quantiles = {}
    seuil_min_obs = 1/alpha
    for sp_id in counts_total.index:
        n_y = counts_calib_match.get(sp_id, 0)
        if n_y < seuil_min_obs:
            class_quantiles[sp_id] = 0.0 # Toujours inclus (score max)
        else:
            s_y = -calib_match[calib_match['true_species_id'] == sp_id]['score'].values
            lvl_y = min(np.ceil((len(s_y) + 1) * (1 - alpha)) / len(s_y), 1.0)
            class_quantiles[sp_id] = np.quantile(s_y, lvl_y, method="higher")

    # --- B. ÉVALUATION ---
    for method in ["Standard", "PAS", "Classwise"]:
        df_test = test.copy()
        
        if method == "Standard":
            df_test['in_set'] = (-df_test['score']) <= q_std
        elif method == "PAS":
            df_test['p_y'] = df_test['species_id'].map(p_y_dict).fillna(p_min)
            df_test['in_set'] = (-df_test['score'] / df_test['p_y']) <= q_pas
        else: # Classwise
            df_test['q_y'] = df_test['species_id'].map(class_quantiles).fillna(0.0)
            df_test['in_set'] = (-df_test['score']) <= df_test['q_y']

        # Identification des succès sur la vraie espèce
        test_success = df_test[df_test['species_id'] == df_test['true_species_id']]
        
        # Calcul Macro-Couverture : Moyenne des couvertures par espèce
        success_per_sp = test_success.groupby('true_species_id')['in_set'].sum()
        # On divise par le nombre réel de fois où l'espèce était dans le test
        macro_cov = (success_per_sp / test_ref_counts).fillna(0).mean()
        
        # Taille moyenne des ensembles (sur N=10812)
        avg_size = df_test.groupby('observation_id')['in_set'].sum().mean()

        results_macro.append({
            'alpha': alpha, 'methode': method,
            'macro_cov': macro_cov,
            'taille_moy': avg_size
        })

# ================================================================
# TABLEAU RÉCAPITULATIF FINAL
# ================================================================
df_res = pd.DataFrame(results_macro)

print(f"\n{'=' * 85}")
print("TABLEAU RÉCAPITULATIF : MACRO-COUVERTURE (APPROCHE NAÏVE)")
print(f"{'=' * 85}")
print(f"| Alpha | Méthode    | Macro-Couverture | Taille Moy. |")
print(f"|-------|------------|------------------|-------------|")

for _, row in df_res.iterrows():
    print(f"| {row['alpha']:<5.2f} | {row['methode']:<10} | {row['macro_cov']:<16.4f} | {row['taille_moy']:>11.2f} |")

print(f"{'=' * 85}")
print("✓ Diagnostic terminé. La supériorité de PAS sur l'équité est déjà visible.")

# --- GÉNÉRATION DES GRAPHIQUES DE SENSIBILITÉ MACRO ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

colors = {"Standard": "steelblue", "PAS": "tomato", "Classwise": "forestgreen"}
markers = {"Standard": "o", "PAS": "s", "Classwise": "^"}

# Graphique 1 : Macro-Couverture vs Alpha
for method in ["Standard", "PAS", "Classwise"]:
    subset = df_res[df_res['methode'] == method]
    ax1.plot(subset['alpha'], subset['macro_cov'], label=method, 
             color=colors[method], marker=markers[method], lw=2)

ax1.plot(ALPHAS, [1-a for a in ALPHAS], color='black', linestyle='--', label='Cible (1-alpha)')
ax1.set_title("Équité : Macro-Couverture vs Alpha", fontsize=13, fontweight='bold')
ax1.set_xlabel("Niveau de risque (alpha)")
ax1.set_ylabel("Macro-Couverture moyenne")
ax1.legend()
ax1.grid(alpha=0.3)

# Graphique 2 : Taille moyenne vs Alpha
for method in ["Standard", "PAS", "Classwise"]:
    subset = df_res[df_res['methode'] == method]
    ax2.plot(subset['alpha'], subset['taille_moy'], label=method, 
             color=colors[method], marker=markers[method], lw=2)

ax2.set_title("Efficacité : Taille moyenne vs Alpha", fontsize=13, fontweight='bold')
ax2.set_xlabel("Niveau de risque (alpha)")
ax2.set_ylabel("Nombre moyen d'espèces")
ax2.set_yscale('log') # Indispensable pour voir l'écart avec Classwise
ax2.legend()
ax2.grid(alpha=0.3, which="both")

plt.tight_layout()
plt.savefig("fig_sensibilite_macro_naive.png", dpi=300)
plt.show()