"""
=============================================================================
Étude 5.4 : Résultats Expérimentaux des Stratégies A et B
=============================================================================
Projet : Pl@ntNet-CP — Prédiction Conformelle
Objectif : Générer les données pour les Tableaux 9 (Strat A) et 10 (Strat B).
           Comparaison Standard CP vs PAS CP sur 3 niveaux d'alpha.
           
            - Stratégie A : Softmax manquant = 0.000
            - Stratégie B : Softmax manquant = 0.001
=============================================================================
"""

import pandas as pd
import numpy as np
import os

# ================================================================
# CONFIGURATION
# ================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PATH_CALIB = os.path.join(SCRIPT_DIR, "expert_calib_50.csv")
PATH_TEST  = os.path.join(SCRIPT_DIR, "expert_test_50.csv")

ALPHAS       = [0.01, 0.05, 0.20]
N_THEORIQUE  = 10812  # 50% de l'échantillon expert total (21624)

# ================================================================
# CHARGEMENT ET PRÉ-TRAITEMENT
# ================================================================
print("=" * 95)
print("GÉNÉRATION DES RÉSULTATS POUR LES TABLEAUX 9 ET 10")
print("=" * 95)

calib = pd.read_csv(PATH_CALIB)
test  = pd.read_csv(PATH_TEST)

# Nettoyage des IDs pour garantir les jointures
for df in [calib, test]:
    df['species_id']      = df['species_id'].astype(str).str.strip()
    df['true_species_id'] = df['true_species_id'].astype(str).str.strip()
    df['observation_id']  = df['observation_id'].astype(str).str.strip()

# 1. Prévalence p_hat(y) sur calibration complète (N=10812)
counts_cal_total = calib.drop_duplicates('observation_id').groupby('true_species_id').size()
p_y_dict = (counts_cal_total / N_THEORIQUE).to_dict()
p_min = counts_cal_total.min() / N_THEORIQUE

# 2. Identification des matches (données présentes)
calib_match = calib[calib['species_id'] == calib['true_species_id']].copy()
n_missing   = N_THEORIQUE - len(calib_match)

# 3. Référence pour la couverture Macro du test
test_ref_counts = test.drop_duplicates('observation_id').groupby('true_species_id').size()

# ================================================================
# FONCTION DE CALCUL DES PERFORMANCES
# ================================================================
def get_table_data(strat_label, p_impute):
    results = []
    
    for alpha in ALPHAS:
        # --- A. CALCUL DES QUANTILES ---
        # Standard
        s_std_match = -calib_match['score'].values
        s_std_miss  = np.full(n_missing, -p_impute)
        s_std_final = np.concatenate([s_std_match, s_std_miss])
        q_std = np.quantile(s_std_final, np.ceil((N_THEORIQUE+1)*(1-alpha))/N_THEORIQUE, method="higher")
        
        # PAS
        calib_match['p_y'] = calib_match['true_species_id'].map(p_y_dict).fillna(p_min)
        s_pas_match = -(calib_match['score'] / calib_match['p_y']).values
        # Pour PAS Stratégie B, on divise par p_min (cas conservateur)
        val_pas_miss = -p_impute / p_min if p_impute > 0 else 0.0
        s_pas_miss  = np.full(n_missing, val_pas_miss)
        s_pas_final = np.concatenate([s_pas_match, s_pas_miss])
        q_pas = np.quantile(s_pas_final, np.ceil((N_THEORIQUE+1)*(1-alpha))/N_THEORIQUE, method="higher")

        # --- B. ÉVALUATION SUR TEST ---
        for method, q in [("Standard", q_std), ("PAS", q_pas)]:
            df_t = test.copy()
            df_t['p_y'] = df_t['species_id'].map(p_y_dict).fillna(p_min)
            
            if method == "Standard":
                df_t['in_set'] = (-df_t['score']) <= q
            else:
                df_t['in_set'] = (-df_t['score'] / df_t['p_y']) <= q
            
            success_mask = (df_t['species_id'] == df_t['true_species_id']) & (df_t['in_set'])
            marg = df_t[success_mask]['observation_id'].nunique() / N_THEORIQUE
            
            success_per_sp = df_t[success_mask].groupby('true_species_id').size()
            macro = (success_per_sp / test_ref_counts).fillna(0).mean()
            size = df_t.groupby('observation_id')['in_set'].sum().mean()
            
            results.append({'alpha': alpha, 'Methode': method, 'Marg.': marg, 'Macro': macro, 'Taille': size})
            
    return pd.DataFrame(results)

# ================================================================
# EXÉCUTION ET AFFICHAGE
# ================================================================
df_a = get_table_data("Stratégie A", 0.000)
df_b = get_table_data("Stratégie B", 0.001)

print(f"\n{'=' * 30} TABLEAU 9 (STRATÉGIE A) {'=' * 30}")
print(df_a.to_string(index=False, formatters={'Marg.': '{:.4f}'.format, 'Macro': '{:.4f}'.format, 'Taille': '{:.2f}'.format}))

print(f"\n{'=' * 30} TABLEAU 10 (STRATÉGIE B) {'=' * 30}")
print(df_b.to_string(index=False, formatters={'Marg.': '{:.4f}'.format, 'Macro': '{:.4f}'.format, 'Taille': '{:.2f}'.format}))

print(f"\n{'=' * 95}")
print("✓ Analyse terminée. Les données correspondent aux structures du rapport.")
print("=" * 95)