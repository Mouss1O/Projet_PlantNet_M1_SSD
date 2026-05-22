"""
=============================================================================
Étude 1 : Couverture Marginale - Approche Naïve (Standard, PAS, Classwise)
=============================================================================
Projet : Pl@ntNet-CP — Prédiction Conformelle
Objectif : Évaluer la couverture globale des trois méthodes sur 4 alphas.
           Approche Naïve : on ignore les données tronquées en calibration.
           CORRECTION : Ajout de min(..., 1.0) pour éviter le plantage Classwise.
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
N_THEO     = 10812 

# ================================================================
# CHARGEMENT ET PRÉPARATION
# ================================================================
print("=" * 95)
print("ÉTUDE DE LA COUVERTURE MARGINALE : TROIS MÉTHODES (APPROCHE NAÏVE)")
print("=" * 95)

calib = pd.read_csv(PATH_CALIB)
test  = pd.read_csv(PATH_TEST)

for df in [calib, test]:
    df['observation_id']  = df['observation_id'].astype(str)
    df['species_id']      = df['species_id'].astype(str)
    df['true_species_id'] = df['true_species_id'].astype(str)

# Prévalence pour PAS
prevalence_counts = calib.drop_duplicates('observation_id').groupby('true_species_id').size()
p_y_dict = (prevalence_counts / N_THEO).to_dict()
p_min = prevalence_counts.min() / N_THEO

# Isole les matches (Naïf)
calib_match = calib[calib['species_id'] == calib['true_species_id']].copy()
counts_calib = calib_match.groupby('true_species_id').size()

# ================================================================
# BOUCLE SUR LES ALPHAS
# ================================================================
results_list = []

for alpha in ALPHAS:
    print(f"\nCalcul pour alpha = {alpha}...")
    
    # --- 1. CALCUL DES SEUILS ---
    n_cal = len(calib_match)
    
    # 1.1 Standard
    s_std_cal = -calib_match['score'].values
    lvl_std = min(np.ceil((n_cal + 1) * (1 - alpha)) / n_cal, 1.0)
    q_std = np.quantile(s_std_cal, lvl_std, method="higher")
    
    # 1.2 PAS
    calib_match['p_y'] = calib_match['true_species_id'].map(p_y_dict).fillna(p_min)
    s_pas_cal = -(calib_match['score'] / calib_match['p_y']).values
    lvl_pas = min(np.ceil((n_cal + 1) * (1 - alpha)) / n_cal, 1.0)
    q_pas = np.quantile(s_pas_cal, lvl_pas, method="higher")
    
    # 1.3 Classwise (Dictionnaire de quantiles par espèce)
    class_quantiles = {}
    seuil_min_obs = 1 / alpha
    
    for sp_id in prevalence_counts.index:
        n_y = counts_calib.get(sp_id, 0)
        if n_y < seuil_min_obs:
            # Règle du blog : Toujours inclure (Seuil maximal possible = 0.0 car s = -p)
            class_quantiles[sp_id] = 0.0 
        else:
            s_y = -calib_match[calib_match['true_species_id'] == sp_id]['score'].values
            # CORRECTION : Ajout de min(..., 1.0) pour éviter le ValueError
            lvl_y = min(np.ceil((len(s_y) + 1) * (1 - alpha)) / len(s_y), 1.0)
            class_quantiles[sp_id] = np.quantile(s_y, lvl_y, method="higher")

    # --- 2. ÉVALUATION SUR TEST ---
    for method_name in ["Standard", "PAS", "Classwise"]:
        temp_test = test.copy()
        
        if method_name == "Standard":
            temp_test['in_set'] = (-temp_test['score']) <= q_std
        elif method_name == "PAS":
            temp_test['p_y'] = temp_test['species_id'].map(p_y_dict).fillna(p_min)
            temp_test['in_set'] = (-temp_test['score'] / temp_test['p_y']) <= q_pas
        else: # Classwise
            temp_test['q_class'] = temp_test['species_id'].map(class_quantiles).fillna(0.0)
            temp_test['in_set'] = (-temp_test['score']) <= temp_test['q_class']
            
        # Métriques
        test_success = temp_test[temp_test['species_id'] == temp_test['true_species_id']]
        nb_succes = test_success['in_set'].sum()
        
        marg_cov = nb_succes / N_THEO
        avg_size = temp_test.groupby('observation_id')['in_set'].sum().mean()
        
        results_list.append({
            'alpha': alpha, 'cible': 1-alpha, 'methode': method_name,
            'cov_marginale': marg_cov, 'taille_moy': avg_size
        })

# ================================================================
# TABLEAU RÉCAPITULATIF FINAL
# ================================================================
df_res = pd.DataFrame(results_list)

print(f"\n{'=' * 95}")
print("TABLEAU RÉCAPITULATIF : COUVERTURE MARGINALE (APPROCHE NAÏVE)")
print(f"{'=' * 95}")
print(f"| Alpha | Cible | Méthode    | Cov. Marginale | Taille Moy. | État             |")
print(f"|-------|-------|------------|----------------|-------------|------------------|")

for _, row in df_res.iterrows():
    etat = "ÉCHEC (Sous-couv)" if row['cov_marginale'] < row['cible'] - 0.005 else "OK"
    print(f"| {row['alpha']:<5.2f} | {row['cible']:<5.2f} | {row['methode']:<10} | "
          f"{row['cov_marginale']:<14.4f} | {row['taille_moy']:<11.2f} | {etat:<16} |")

print(f"{'=' * 95}")

# --- GÉNÉRATION DES GRAPHIQUES DE SENSIBILITÉ ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Couleurs pour la cohérence du rapport
colors = {"Standard": "steelblue", "PAS": "tomato", "Classwise": "forestgreen"}
markers = {"Standard": "o", "PAS": "s", "Classwise": "^"}

# Graphique 1 : Couverture vs Alpha
for method in ["Standard", "PAS", "Classwise"]:
    subset = df_res[df_res['methode'] == method]
    ax1.plot(subset['alpha'], subset['cov_marginale'], label=method, 
             color=colors[method], marker=markers[method], lw=2)

# Ajout de la cible théorique (Diagonale 1-alpha)
ax1.plot(ALPHAS, [1-a for a in ALPHAS], color='black', linestyle='--', label='Cible (1-alpha)')

ax1.set_title("Fiabilité : Couverture Marginale vs Alpha", fontsize=13, fontweight='bold')
ax1.set_xlabel("Niveau de risque (alpha)")
ax1.set_ylabel("Couverture observée")
ax1.legend()
ax1.grid(alpha=0.3)

# Graphique 2 : Taille moyenne vs Alpha
for method in ["Standard", "PAS", "Classwise"]:
    subset = df_res[df_res['methode'] == method]
    ax2.plot(subset['alpha'], subset['taille_moy'], label=method, 
             color=colors[method], marker=markers[method], lw=2)

ax2.set_title("Efficacité : Taille moyenne vs Alpha", fontsize=13, fontweight='bold')
ax2.set_xlabel("Niveau de risque (alpha)")
ax2.set_ylabel("Nombre moyen d'espèces dans C(x)")
ax2.set_yscale('log') # Échelle log car Classwise est très élevé
ax2.legend()
ax2.grid(alpha=0.3, which="both")

plt.tight_layout()
plt.savefig("fig_sensibilite_naive.png", dpi=300)
plt.show()