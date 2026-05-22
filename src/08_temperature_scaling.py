"""
=============================================================================
Étude 5.5 : Impact de la Température (T) — Stratégies A et B
=============================================================================
Projet : Pl@ntNet-CP — Prédiction Conformelle
Objectif : Évaluer l'impact du Temperature Scaling sur l'efficacité (taille)
           et la fiabilité (couverture). 
           Comparaison des stratégies A (p=0.000) et B (p=0.001).
           Affichage de deux courbes comparatives : Standard vs PAS.

Entrées :
    - expert_calib_50.csv 
    - expert_test_50.csv
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
DIR_FIGURES = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(DIR_FIGURES, exist_ok=True)

ALPHA        = 0.05
N_THEORIQUE  = 10812 
TEMPERATURES = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]

# ================================================================
# CHARGEMENT ET PRÉ-TRAITEMENT
# ================================================================
print("=" * 95)
print("ANALYSE DE TEMPÉRATURE : STRATÉGIES A ET B")
print("=" * 95)

calib = pd.read_csv(os.path.join(SCRIPT_DIR, "expert_calib_50.csv"))
test  = pd.read_csv(os.path.join(SCRIPT_DIR, "expert_test_50.csv"))

for df in [calib, test]:
    df['species_id']      = df['species_id'].astype(str).str.strip()
    df['true_species_id'] = df['true_species_id'].astype(str).str.strip()

# Prévalence p_hat(y) sur calibration complète
counts_total = calib.drop_duplicates('observation_id').groupby('true_species_id').size()
p_y_dict = (counts_total / N_THEORIQUE).to_dict()
p_min = counts_total.min() / N_THEORIQUE

# Identification des matches
calib_match = calib[calib['species_id'] == calib['true_species_id']].copy()
n_missing   = N_THEORIQUE - len(calib_match)

# Référence pour la couverture Macro du test
test_ref_counts = test.drop_duplicates('observation_id').groupby('true_species_id').size()

# ================================================================
# FONCTION DE CALCUL
# ================================================================
def run_study(strat_name, p_val):
    results = []
    for T in TEMPERATURES:
        # 1. Quantiles corrigés
        # Standard
        s_std_match = -(calib_match['score']**(1/T)).values
        s_std_miss  = np.full(n_missing, -(p_val**(1/T)))
        s_std_final = np.concatenate([s_std_match, s_std_miss])
        q_std = np.quantile(s_std_final, np.ceil((N_THEORIQUE+1)*(1-ALPHA))/N_THEORIQUE, method="higher")
        
        # PAS
        calib_match['p_y'] = calib_match['true_species_id'].map(p_y_dict).fillna(p_min)
        s_pas_match = -( (calib_match['score']**(1/T)) / calib_match['p_y'] ).values
        s_pas_miss  = np.full(n_missing, -(p_val**(1/T)) / p_min)
        s_pas_final = np.concatenate([s_pas_match, s_pas_miss])
        q_pas = np.quantile(s_pas_final, np.ceil((N_THEORIQUE+1)*(1-ALPHA))/N_THEORIQUE, method="higher")

        # 2. Évaluation
        for method, q in [("Standard", q_std), ("PAS", q_pas)]:
            df_t = test.copy()
            df_t['p_y'] = df_t['species_id'].map(p_y_dict).fillna(p_min)
            
            if method == "Standard":
                df_t['in_set'] = (-(df_t['score']**(1/T))) <= q
            else:
                df_t['in_set'] = (-( (df_t['score']**(1/T)) / df_t['p_y'] )) <= q
            
            test_success = df_t[df_t['species_id'] == df_t['true_species_id']]
            marg = test_success['in_set'].sum() / N_THEORIQUE
            macro = (test_success.groupby('true_species_id')['in_set'].sum() / test_ref_counts).fillna(0).mean()
            size = df_t.groupby('observation_id')['in_set'].sum().mean()
            
            results.append({'T': T, 'strat': strat_name, 'method': method, 'marg': marg, 'macro': macro, 'size': size})
    return pd.DataFrame(results)

# ================================================================
# EXÉCUTION
# ================================================================
print(f"Calcul des performances (Stratégies A et B)...")
df_a = run_study("A", 0.000)
df_b = run_study("B", 0.001)
df_all = pd.concat([df_a, df_b])

# ================================================================
# GÉNÉRATION DES COURBES (1, 2)
# ================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# --- GRAPHIQUE 1 : STANDARD CP ---
data_std_a = df_all[(df_all['method']=='Standard') & (df_all['strat']=='A')]
data_std_b = df_all[(df_all['method']=='Standard') & (df_all['strat']=='B')]

ax1.plot(TEMPERATURES, data_std_a['size'], 'o-', label='Taille (Strat A)', color='steelblue', lw=2)
ax1.plot(TEMPERATURES, data_std_b['size'], 's--', label='Taille (Strat B)', color='skyblue', lw=2)
# Axe secondaire pour la couverture
ax1_tw = ax1.twinx()
ax1_tw.plot(TEMPERATURES, data_std_a['marg'], 'x-', color='red', alpha=0.6, label='Couv. Marginale')
ax1_tw.axhline(0.95, color='black', ls=':', label='Cible 95%')

ax1.set_title("Standard CP : Impact de T", fontsize=14, fontweight='bold')
ax1.set_xlabel("Température (T)"); ax1.set_ylabel("Taille moyenne des ensembles"); ax1_tw.set_ylabel("Couverture")
ax1.legend(loc='upper left'); ax1_tw.legend(loc='upper right'); ax1.grid(alpha=0.3)

# --- GRAPHIQUE 2 : PAS CP ---
data_pas_a = df_all[(df_all['method']=='PAS') & (df_all['strat']=='A')]
data_pas_b = df_all[(df_all['method']=='PAS') & (df_all['strat']=='B')]

ax2.plot(TEMPERATURES, data_pas_a['size'], 'o-', label='Taille (Strat A)', color='tomato', lw=2)
ax2.plot(TEMPERATURES, data_pas_b['size'], 's--', label='Taille (Strat B)', color='coral', lw=2)
# Axe secondaire pour la couverture
ax2_tw = ax2.twinx()
ax2_tw.plot(TEMPERATURES, data_pas_a['macro'], 'x-', color='darkred', alpha=0.6, label='Couv. Macro')
ax2_tw.axhline(0.95, color='black', ls=':', label='Cible 95%')

ax2.set_title("PAS CP : Impact de T", fontsize=14, fontweight='bold')
ax2.set_xlabel("Température (T)"); ax2.set_ylabel("Taille moyenne des ensembles"); ax2_tw.set_ylabel("Couverture")
ax2.legend(loc='upper left'); ax2_tw.legend(loc='upper right'); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(DIR_FIGURES, "fig_temperature_scaling_comparison.png"), dpi=200)
plt.show()

# ================================================================
# TABLEAU FINAL
# ================================================================
print(f"\n{'=' * 95}")
print("RÉSUMÉ POUR LE RAPPORT")
print(f"{'=' * 95}")
summary = df_all[df_all['T'].isin([0.5, 1.0, 2.0])]
print(summary.to_string(index=False))