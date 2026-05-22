"""
=============================================================================
Étude 2 : Couverture Conditionnelle — Visualisation (Naïve)
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# CONFIGURATION
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_FIGURES = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(DIR_FIGURES, exist_ok=True)

ALPHAS = [0.01, 0.05, 0.10, 0.20]
N_THEO = 10812 

# CHARGEMENT
calib = pd.read_csv(os.path.join(SCRIPT_DIR, "expert_calib_50.csv"))
test  = pd.read_csv(os.path.join(SCRIPT_DIR, "expert_test_50.csv"))

for df in [calib, test]:
    df['species_id'] = df['species_id'].astype(str).str.strip()
    df['true_species_id'] = df['true_species_id'].astype(str).str.strip()

# Calcul de la prévalence sur calibration (Nécessaire pour PAS et le Graphique 2)
prevalence_counts = calib.drop_duplicates('observation_id').groupby('true_species_id').size()
p_y_dict = (prevalence_counts / N_THEO).to_dict()
p_min = prevalence_counts.min() / N_THEO

# Matchs Naïfs
calib_match = calib[calib['species_id'] == calib['true_species_id']].copy()
counts_calib_match = calib_match.groupby('true_species_id').size()

results_cond = []

for alpha in ALPHAS:
    print(f"Calcul pour alpha = {alpha}...")
    
    # 1. CALCUL DES QUANTILES
    n_cal = len(calib_match)
    lvl = min(np.ceil((n_cal + 1) * (1 - alpha)) / n_cal, 1.0)
    
    q_std = np.quantile(-calib_match['score'].values, lvl, method="higher")
    
    calib_match['p_y'] = calib_match['true_species_id'].map(p_y_dict).fillna(p_min)
    q_pas = np.quantile(-(calib_match['score'] / calib_match['p_y']).values, lvl, method="higher")

    class_quantiles = {}
    seuil_min_obs = 1/alpha
    for sp_id in prevalence_counts.index:
        n_y = counts_calib_match.get(sp_id, 0)
        if n_y < seuil_min_obs:
            class_quantiles[sp_id] = 0.0
        else:
            s_y = -calib_match[calib_match['true_species_id'] == sp_id]['score'].values
            lvl_y = min(np.ceil((len(s_y) + 1) * (1 - alpha)) / len(s_y), 1.0)
            class_quantiles[sp_id] = np.quantile(s_y, lvl_y, method="higher")

    # 2. ÉVALUATION (Stockage pour les graphiques si alpha == 0.05)
    cov_data_005 = {}

    for method in ["Standard", "PAS", "Classwise"]:
        df_test = test.copy()
        if method == "Standard":
            df_test['in_set'] = (-df_test['score']) <= q_std
        elif method == "PAS":
            df_test['p_y'] = df_test['species_id'].map(p_y_dict).fillna(p_min)
            df_test['in_set'] = (-df_test['score'] / df_test['p_y']) <= q_pas
        else:
            df_test['q_y'] = df_test['species_id'].map(class_quantiles).fillna(0.0)
            df_test['in_set'] = (-df_test['score']) <= df_test['q_y']

        test_true = df_test[df_test['species_id'] == df_test['true_species_id']]
        cov_per_sp = test_true.groupby('true_species_id')['in_set'].mean()
        
        # On garde les données pour le graphique de alpha = 0.05
        if alpha == 0.05:
            cov_data_005[method] = cov_per_sp

        results_cond.append({'alpha': alpha, 'methode': method, 'std_dev': cov_per_sp.std(), 'pct_null': (cov_per_sp == 0).mean()*100})

    # 3. DESSIN DES GRAPHIQUES (Seulement pour alpha = 0.05)
    if alpha == 0.05:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))

        # Histogramme
        ax1.hist(cov_data_005['Standard'], bins=20, alpha=0.5, label='Standard CP', color='steelblue')
        ax1.hist(cov_data_005['PAS'], bins=20, alpha=0.5, label='PAS CP', color='tomato')
        ax1.axvline(0.95, color='black', linestyle='--', label='Cible 95%')
        ax1.set_title("Distribution de la couverture par espèce", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Taux de couverture observé"); ax1.set_ylabel("Nombre d'espèces"); ax1.legend()

        # Scatter plot vs Prévalence
        plot_df = pd.DataFrame({
            'n_images': prevalence_counts,
            'cov_std': cov_data_005['Standard'],
            'cov_pas': cov_data_005['PAS']
        }).fillna(0).sort_values('n_images')

        ax2.scatter(plot_df['n_images'], plot_df['cov_std'], alpha=0.2, color='steelblue', s=10)
        ax2.scatter(plot_df['n_images'], plot_df['cov_pas'], alpha=0.2, color='tomato', s=10)
        # Courbes de tendance
        ax2.plot(plot_df['n_images'], plot_df['cov_std'].rolling(100, min_periods=1).mean(), color='blue', lw=2, label='Tendance Standard')
        ax2.plot(plot_df['n_images'], plot_df['cov_pas'].rolling(100, min_periods=1).mean(), color='red', lw=2, label='Tendance PAS')
        
        ax2.set_xscale('log'); ax2.axhline(0.95, color='black', linestyle='--')
        ax2.set_title("Couverture vs Prévalence (Axe X Log)", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Nombre d'images de l'espèce"); ax2.set_ylabel("Couverture"); ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(DIR_FIGURES, "fig_conditional_naive_alpha005.png"))
        plt.show()

# Affichage du tableau final
print(pd.DataFrame(results_cond).to_string(index=False))