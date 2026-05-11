"""
Phase 3 – Traitement des observations manquantes
Comparaison des stratégies A, B, C
Auteur : Firda
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

ALPHA = 0.05

# ============================================================
# 1. CHARGEMENT
# ============================================================
print("Chargement des données...")
calib = pd.read_csv("fichiers_utilises/calibration_with_scores_long.csv")
test  = pd.read_csv("fichiers_utilises/test_with_scores_long.csv")
scores = pd.read_csv("fichiers_utilises/ai_scores_all.csv")

# ============================================================
# 2. FONCTION PRINCIPALE (détection correcte des manquantes)
# ============================================================
def evaluer_strategie(calib_df, test_df, scores_df, alpha, strategy='A'):
    """
    strategy : 'A' -> S=1.0
               'B' -> S=0.999
               'C' -> exclusion + correction du quantile
    """
    # ---- Identifier les observations manquantes (bonne espèce absente des scores) ----
    # Tous les couples (observation_id, ground_truth_val) de la calibration
    calib_gt = calib_df[['observation_id', 'ground_truth_val']].drop_duplicates()
    # Couples présents dans les scores (observation_id, spicies_id) -> renommer spicies_id en ground_truth_val
    scores_pairs = scores_df[['observation_id', 'spicies_id']].drop_duplicates()
    scores_pairs = scores_pairs.rename(columns={'spicies_id': 'ground_truth_val'})
    # Fusion : ceux qui ont un score pour la vraie espèce
    present = calib_gt.merge(scores_pairs, on=['observation_id', 'ground_truth_val'], how='inner')
    manquantes = calib_gt[~calib_gt['observation_id'].isin(present['observation_id'])].copy()
    print(f"      [Strat. {strategy}] Obs. manquantes détectées : {len(manquantes)}")

    # ---- Récupérer les scores pour les observations présentes ----
    calib_scores = scores_df.merge(calib_df[['observation_id', 'ground_truth_val']],
                                   on='observation_id')
    calib_true = calib_scores[calib_scores['spicies_id'] == calib_scores['ground_truth_val']].copy()
    calib_true['non_conformity'] = 1 - calib_true['score']

    # ---- Construire la série complète des non-conformités selon la stratégie ----
    if strategy in ['A', 'B']:
        # Ajouter les manquantes avec une valeur fixe
        if strategy == 'A':
            manquantes['non_conformity'] = 1.0
        else:  # B
            manquantes['non_conformity'] = 0.999
        manquantes = manquantes[['observation_id', 'non_conformity']]
        non_conf = pd.concat([calib_true[['observation_id', 'non_conformity']], manquantes],
                             ignore_index=True)
        n_total = len(non_conf)
        quantile_level = min((1 - alpha) * (1 + 1 / n_total), 1.0)
        q_hat = np.quantile(non_conf['non_conformity'], quantile_level, method='higher')
        seuil_score = 1 - q_hat

    else:  # strategy 'C'
        # Exclure les manquantes, mais corriger le niveau du quantile
        non_conf = calib_true[['observation_id', 'non_conformity']].copy()
        n_ok = len(non_conf)                     # observations présentes
        n_obs_calib = calib_df['observation_id'].nunique()   # total (y compris manquantes)
        quantile_level = ((n_obs_calib + 1) * (1 - alpha)) / n_ok
        quantile_level = min(quantile_level, 1.0)
        q_hat = np.quantile(non_conf['non_conformity'], quantile_level, method='higher')
        seuil_score = 1 - q_hat

    # ---- Construction des ensembles sur le test ----
    ground_truth = test_df[['observation_id', 'ground_truth_val']].drop_duplicates()
    test_scores = scores_df.merge(ground_truth, on='observation_id')

    sets_above = test_scores[test_scores['score'] >= seuil_score]
    sets = (sets_above.groupby('observation_id')
            .agg(gardees=('spicies_id', list),
                 vraie_espece=('ground_truth_val', 'first'))
            .reset_index())

    # Observations sans aucune espèce retenue -> argmax
    no_cov = ground_truth[~ground_truth['observation_id'].isin(sets['observation_id'])]
    if len(no_cov) > 0:
        fallback = (test_scores[test_scores['observation_id'].isin(no_cov['observation_id'])]
                    .sort_values('score', ascending=False)
                    .groupby('observation_id')
                    .agg(gardees=('spicies_id', lambda x: [x.iloc[0]]),
                         vraie_espece=('ground_truth_val', 'first'))
                    .reset_index())
        sets = pd.concat([sets, fallback], ignore_index=True)

    sets['taille'] = sets['gardees'].str.len()
    sets['couvert'] = sets.apply(lambda r: r['vraie_espece'] in r['gardees'], axis=1)

    # ---- Métriques ----
    cov_marg = sets['couvert'].mean()
    cov_par_espece = sets.groupby('vraie_espece')['couvert'].mean()
    cov_macro = cov_par_espece.mean()
    taille_moy = sets['taille'].mean()

    return {
        'q_hat': q_hat,
        'seuil_score': seuil_score,
        'coverage_marginale': cov_marg,
        'coverage_macro': cov_macro,
        'taille_moyenne': taille_moy,
        'n_calib_used': len(non_conf) if strategy in ['A', 'B'] else n_ok,
        'quantile_level': quantile_level
    }

# ============================================================
# 3. LANCER LES 3 STRATÉGIES
# ============================================================
resultats = {}
for strat in ['A', 'B', 'C']:
    print(f"\n--- Stratégie {strat} ---")
    res = evaluer_strategie(calib, test, scores, ALPHA, strategy=strat)
    resultats[strat] = res
    print(f"q_hat = {res['q_hat']:.6f}")
    print(f"Couverture marginale = {res['coverage_marginale']:.3f}")
    print(f"Couverture macro = {res['coverage_macro']:.3f}")
    print(f"Taille moyenne = {res['taille_moyenne']:.2f}")

# ============================================================
# 4. TABLEAU RÉCAPITULATIF
# ============================================================
df_results = pd.DataFrame(resultats).T
df_results.to_csv("results/resultats_strategies_manquantes.csv")
print("\n✅ Résultats sauvegardés dans results/resultats_strategies_manquantes.csv")
print(df_results[['q_hat', 'coverage_marginale', 'coverage_macro', 'taille_moyenne']])

# ============================================================
# 5. VISUALISATIONS
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Barplot couverture
ax = axes[0]
strategies = list(resultats.keys())
cov_marg = [resultats[s]['coverage_marginale'] for s in strategies]
cov_macro = [resultats[s]['coverage_macro'] for s in strategies]
x = np.arange(len(strategies))
width = 0.35
ax.bar(x - width/2, cov_marg, width, label='Marginale', color='steelblue')
ax.bar(x + width/2, cov_macro, width, label='Macro', color='coral')
ax.axhline(y=0.95, color='red', linestyle='--', label='Objectif 95%')
ax.set_xticks(x)
ax.set_xticklabels([f'Strat. {s}' for s in strategies])
ax.set_ylabel('Couverture')
ax.set_title('Comparaison des stratégies')
ax.legend()

# Barplot taille moyenne
ax = axes[1]
taille = [resultats[s]['taille_moyenne'] for s in strategies]
ax.bar(strategies, taille, color='lightgreen', edgecolor='black')
ax.set_ylabel('Taille moyenne des ensembles')
ax.set_title('Efficacité (plus petite = meilleure)')
plt.tight_layout()
plt.savefig('figures/comparaison_strategies.png', dpi=150)
plt.close()
print("\n📊 Graphique sauvegardé : figures/comparaison_strategies.png")

# ============================================================
# 6. DISTRIBUTION DES NON-CONFORMITÉS (mutualisé)
# ============================================================
# Pré-calcul commun
calib_gt = calib[['observation_id', 'ground_truth_val']].drop_duplicates()
scores_pairs = scores[['observation_id', 'spicies_id']].drop_duplicates()
scores_pairs = scores_pairs.rename(columns={'spicies_id': 'ground_truth_val'})
present = calib_gt.merge(scores_pairs, on=['observation_id', 'ground_truth_val'], how='inner')
manquantes_base = calib_gt[~calib_gt['observation_id'].isin(present['observation_id'])].copy()

calib_scores_base = scores.merge(calib[['observation_id', 'ground_truth_val']], on='observation_id')
calib_true_base = calib_scores_base[
    calib_scores_base['spicies_id'] == calib_scores_base['ground_truth_val']
].copy()
calib_true_base['non_conformity'] = 1 - calib_true_base['score']

plt.figure(figsize=(10, 6))
for s, col in zip(['A', 'B', 'C'], ['blue', 'green', 'red']):
    if s in ['A', 'B']:
        manq = manquantes_base.copy()
        manq['non_conformity'] = 1.0 if s == 'A' else 0.999
        non_conf = pd.concat([calib_true_base['non_conformity'], manq['non_conformity']], ignore_index=True)
    else:
        non_conf = calib_true_base['non_conformity']
    plt.hist(non_conf, bins=50, alpha=0.5, label=f'Strat. {s}', color=col, density=True)

plt.xlabel('Score de non‑conformité S')
plt.ylabel('Densité')
plt.title('Distribution des scores de non‑conformité pour chaque stratégie')
plt.legend()
plt.tight_layout()
plt.savefig('figures/distribution_strategies.png', dpi=150)
plt.close()
print("📊 Second graphique : figures/distribution_strategies.png")