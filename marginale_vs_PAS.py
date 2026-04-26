"""
PHASE 2 COMPLETE - Split Conformal Prediction
Analyse de la performance : couverture marginale vs macro
Comparaison : Standard CP vs PAS (Species-Aware Score)
Auteur: Firda
Date: Avril 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Créer les dossiers
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# Style des graphiques
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

print("=" * 70)
print("PHASE 2 COMPLETE - Split Conformal Prediction")
print("Analyse : Couverture marginale vs Couverture macro")
print("Comparaison : Standard CP vs PAS")
print("=" * 70)

# ============================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================

print("\n📂 1. Chargement des données...")

calib = pd.read_csv("fichiers_utilises/calibration_with_scores_long.csv")
test = pd.read_csv("fichiers_utilises/test_with_scores_long.csv")
scores = pd.read_csv("fichiers_utilises/ai_scores_all.csv")
obs_expert = pd.read_csv("fichiers_utilises/observations_experts.csv")

print(f"   ✓ Calibration    : {len(calib):,} lignes")
print(f"   ✓ Test           : {len(test):,} lignes")
print(f"   ✓ Scores         : {len(scores):,} lignes")
print(f"   ✓ Obs. expert    : {len(obs_expert):,} lignes")

# ============================================================
# 2. SCORES DE NON-CONFORMITÉ (calibration complète)
# ============================================================

print("\n📊 2. Calcul des scores de non-conformité (calibration complète)...")

calib_true = calib[calib['spicies_id'] == calib['ground_truth_val']].copy()
calib_true['non_conformity'] = 1 - calib_true['score']

n_calib = len(calib_true)
alpha = 0.05

print(f"   ✓ Observations calibration valides : {n_calib:,}")
print(f"   ✓ Non-conformité moyenne           : {calib_true['non_conformity'].mean():.4f}")

# ============================================================
# 3. SEUIL STANDARD CP
# ============================================================

print("\n📊 3. Calcul du seuil Standard CP...")

quantile_level = (1 - alpha) * (1 + 1 / n_calib)
q_hat = np.quantile(calib_true['non_conformity'], quantile_level, method='higher')
seuil_score = 1 - q_hat

print(f"   ✓ Quantile level    : {quantile_level:.6f}")
print(f"   ✓ q_hat             : {q_hat:.6f}")
print(f"   ✓ Seuil Standard CP : {seuil_score:.6f}")

# ============================================================
# 4. SEUILS PAS (vectorisé par groupby, calibration complète)
# ============================================================

print("\n📊 4. Calcul des seuils PAS (vectorisé, calibration complète)...")

MIN_OBS_PAS = 10  # minimum d'observations pour un seuil stable par espèce

def calcul_seuil_pas(group):
    n = len(group)
    if n < MIN_OBS_PAS:
        return np.nan  # repli sur seuil global
    ql = min((1 - alpha) * (1 + 1 / n), 1.0)
    return 1 - np.quantile(group['non_conformity'], ql, method='higher')

pas_seuils_series = (
    calib_true
    .groupby('ground_truth_val')
    .apply(calcul_seuil_pas)
    .dropna()
)
pas_seuils = pas_seuils_series.to_dict()

n_especes_pas   = len(pas_seuils)
n_especes_repli = calib_true['ground_truth_val'].nunique() - n_especes_pas

print(f"   ✓ Seuils PAS calculés : {n_especes_pas} espèces (≥ {MIN_OBS_PAS} obs)")
print(f"   ✓ Repli seuil global  : {n_especes_repli} espèces (< {MIN_OBS_PAS} obs)")
print(f"   ✓ Seuil PAS médian    : {pas_seuils_series.median():.4f}")
print(f"   ✓ Seuil PAS min       : {pas_seuils_series.min():.4f}")
print(f"   ✓ Seuil PAS max       : {pas_seuils_series.max():.4f}")

# ============================================================
# 5. CONSTRUCTION DES ENSEMBLES (vectorisé)
# ============================================================

print("\n🔧 5. Construction des ensembles de prédiction...")

ground_truth  = test[['observation_id', 'ground_truth_val']].drop_duplicates()
scores_merged = scores.merge(ground_truth, on='observation_id')

def construire_ensembles(scores_df, seuil_fixe=None, seuils_par_espece=None, seuil_global=None):
    """
    Construit les ensembles de prédiction conformal.
      - seuil_fixe        : seuil unique (Standard CP)
      - seuils_par_espece : dict espèce -> seuil (PAS)
      - seuil_global      : seuil de repli si espèce absente de la calibration
    """
    df = scores_df.copy()

    if seuil_fixe is not None:
        scores_above = df[df['score'] >= seuil_fixe]
    else:
        df['seuil_pas'] = df['ground_truth_val'].map(seuils_par_espece).fillna(seuil_global)
        scores_above    = df[df['score'] >= df['seuil_pas']]

    sets_above = (
        scores_above
        .groupby('observation_id')
        .agg(
            gardees      = ('spicies_id',       list),
            vraie_espece = ('ground_truth_val', 'first')
        )
        .reset_index()
    )

    # Repli : observations sans aucune espèce retenue → garder la meilleure
    no_cov = ground_truth[
        ~ground_truth['observation_id'].isin(sets_above['observation_id'])
    ]
    if len(no_cov) > 0:
        fallback = (
            df[df['observation_id'].isin(no_cov['observation_id'])]
            .sort_values('score', ascending=False)
            .groupby('observation_id')
            .agg(
                gardees      = ('spicies_id',       lambda x: [x.iloc[0]]),
                vraie_espece = ('ground_truth_val', 'first')
            )
            .reset_index()
        )
        sets_above = pd.concat([sets_above, fallback], ignore_index=True)

    sets_above['taille']  = sets_above['gardees'].str.len()
    sets_above['couvert'] = sets_above.apply(
        lambda r: r['vraie_espece'] in r['gardees'], axis=1
    )
    return sets_above[['observation_id', 'vraie_espece', 'taille', 'couvert']]


# Standard CP
df_standard = construire_ensembles(scores_merged, seuil_fixe=seuil_score)

# PAS
df_pas = construire_ensembles(
    scores_merged,
    seuils_par_espece=pas_seuils,
    seuil_global=seuil_score
)

print(f"   ✓ Standard CP : {len(df_standard):,} ensembles construits")
print(f"   ✓ PAS         : {len(df_pas):,} ensembles construits")

# ============================================================
# 6. MÉTRIQUES
# ============================================================

print("\n📈 6. Calcul des métriques...")

def calculer_metriques(df, nom):
    cov_marg       = df['couvert'].mean()
    cov_par_espece = df.groupby('vraie_espece')['couvert'].agg(['mean', 'count'])
    cov_par_espece.columns = ['couverture', 'n_observations']
    cov_macro         = cov_par_espece['couverture'].mean()
    taille_moy        = df['taille'].mean()
    taille_med        = df['taille'].median()
    taille_std        = df['taille'].std()
    pct_singleton     = (df['taille'] == 1).mean() * 100
    rares             = cov_par_espece[cov_par_espece['n_observations'] < 5]
    communes          = cov_par_espece[cov_par_espece['n_observations'] >= 5]
    communes_beaucoup = cov_par_espece[cov_par_espece['n_observations'] >= 20]

    obj_marg = '✅' if cov_marg  >= 0.95 else '❌'
    obj_mac  = '✅' if cov_macro >= 0.95 else '❌'

    print(f"\n   [{nom}]")
    print(f"   Couverture marginale              : {cov_marg:.3f} ({cov_marg*100:.1f}%)  {obj_marg}")
    print(f"   Couverture macro                  : {cov_macro:.3f} ({cov_macro*100:.1f}%)  {obj_mac}")
    print(f"   Écart (macro - marginale)         : {cov_macro - cov_marg:+.3f}")
    print(f"   Taille moyenne                    : {taille_moy:.2f}")
    print(f"   Taille médiane                    : {taille_med:.0f}")
    print(f"   Écart-type tailles                : {taille_std:.2f}")
    print(f"   Singletons                        : {pct_singleton:.1f}%")
    print(f"   Espèces rares (< 5 obs)           : {len(rares)} — couverture {rares['couverture'].mean():.3f}")
    print(f"   Espèces communes (≥ 5 obs)        : {len(communes)} — couverture {communes['couverture'].mean():.3f}")
    print(f"   Espèces très communes (≥ 20 obs)  : {len(communes_beaucoup)} — couverture {communes_beaucoup['couverture'].mean():.3f}")

    return {
        'modele':              nom,
        'coverage_marginale':  cov_marg,
        'coverage_macro':      cov_macro,
        'taille_moyenne':      taille_moy,
        'taille_mediane':      taille_med,
        'taille_std':          taille_std,
        'pct_singleton':       pct_singleton,
        'coverage_rares':      rares['couverture'].mean() if len(rares) > 0 else np.nan,
        'coverage_communes':   communes['couverture'].mean(),
        'nb_especes_rares':    len(rares),
        'nb_especes_communes': len(communes),
        'cov_par_espece':      cov_par_espece
    }

print("\n" + "=" * 70)
print("RÉSULTATS PRINCIPAUX")
print("=" * 70)

m_std = calculer_metriques(df_standard, "Standard CP")
m_pas = calculer_metriques(df_pas,      "PAS")

# ============================================================
# 7. DISTRIBUTION DES TAILLES (Standard CP)
# ============================================================

print("\n📊 7. Distribution des tailles (Standard CP)...")
dist_taille = df_standard['taille'].value_counts().sort_index()
print("   Taille | Observations | Pourcentage")
print("   " + "-" * 40)
for taille, count in dist_taille.items():
    pct   = count / len(df_standard) * 100
    barre = "█" * int(pct / 2)
    print(f"     {taille:2d}    | {count:6,}     | {pct:5.1f}% {barre}")

# ============================================================
# 8. GRAPHIQUES
# ============================================================

print("\n📊 8. Génération des graphiques...")

cpe_std = m_std['cov_par_espece']
cpe_pas = m_pas['cov_par_espece']

# Figure 1 : Couverture marginale vs macro — Standard vs PAS
fig, ax = plt.subplots(figsize=(10, 6))
x     = np.arange(2)
width = 0.35
bars1 = ax.bar(x - width/2,
               [m_std['coverage_marginale'], m_std['coverage_macro']],
               width, label='Standard CP', color='steelblue', edgecolor='black')
bars2 = ax.bar(x + width/2,
               [m_pas['coverage_marginale'], m_pas['coverage_macro']],
               width, label='PAS', color='coral', edgecolor='black')
ax.axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='Objectif 95%')
ax.set_xticks(x)
ax.set_xticklabels(['Couverture\nmarginale', 'Couverture\nmacro'], fontsize=12)
ax.set_ylabel('Taux de couverture', fontsize=12)
ax.set_title('Comparaison Standard CP vs PAS\n(Couverture marginale et macro)', fontsize=14)
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(axis='y', alpha=0.3)
for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom',
            fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/01_standard_vs_pas_couverture.png', dpi=150)
plt.close()

# Figure 2 : Histogramme des couvertures par espèce — Standard vs PAS
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
for ax, (cpe, nom, color) in zip(axes, [
        (cpe_std, 'Standard CP', 'steelblue'),
        (cpe_pas, 'PAS',         'coral')]):
    ax.hist(cpe['couverture'], bins=25, edgecolor='black', alpha=0.7, color=color)
    ax.axvline(x=0.95, color='red',   linestyle='--', linewidth=2, label='Objectif 95%')
    ax.axvline(x=cpe['couverture'].mean(), color='green', linestyle='-', linewidth=2,
               label=f'Macro = {cpe["couverture"].mean():.3f}')
    ax.set_xlabel('Couverture par espèce', fontsize=12)
    ax.set_ylabel("Nombre d'espèces", fontsize=12)
    ax.set_title(f'Distribution des couvertures — {nom}', fontsize=13)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
plt.suptitle('Couverture par espèce : Standard CP vs PAS', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/02_histogramme_couverture_par_espece.png', dpi=150)
plt.close()

# Figure 3 : Couverture vs rareté (scatter) — Standard vs PAS
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
for ax, (cpe, nom, color) in zip(axes, [
        (cpe_std, 'Standard CP', 'steelblue'),
        (cpe_pas, 'PAS',         'coral')]):
    ax.scatter(cpe['n_observations'], cpe['couverture'],
               alpha=0.5, c=color, edgecolors='white', s=50)
    ax.axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='Objectif 95%')
    ax.set_xscale('log')
    ax.set_xlabel("Nombre d'observations par espèce (log)", fontsize=11)
    ax.set_ylabel('Couverture', fontsize=11)
    ax.set_title(f'Couverture vs Rareté — {nom}', fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)
plt.suptitle("Couverture en fonction de la rareté de l'espèce", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/03_couverture_vs_rarete.png', dpi=150)
plt.close()

# Figure 4 : Boxplot rares vs communes — Standard vs PAS
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
for ax, (cpe, nom) in zip(axes, [(cpe_std, 'Standard CP'), (cpe_pas, 'PAS')]):
    data_box = [
        cpe[cpe['n_observations'] < 5]['couverture'].dropna(),
        cpe[cpe['n_observations'] >= 5]['couverture'].dropna()
    ]
    box = ax.boxplot(data_box,
                     tick_labels=['Rares\n(< 5 obs)', 'Communes\n(≥ 5 obs)'],
                     patch_artist=True, widths=0.6)
    for patch, color in zip(box['boxes'], ['salmon', 'lightgreen']):
        patch.set_facecolor(color)
    ax.axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='Objectif 95%')
    ax.set_ylabel('Couverture', fontsize=12)
    ax.set_title(f'Rares vs Communes — {nom}', fontsize=13)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
plt.suptitle('Comparaison espèces rares vs communes : Standard CP vs PAS',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/04_boxplot_rares_vs_communes.png', dpi=150)
plt.close()

# Figure 5 : Distribution des tailles d'ensembles — Standard vs PAS
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, (df, nom, color) in zip(axes, [
        (df_standard, 'Standard CP', 'steelblue'),
        (df_pas,      'PAS',         'coral')]):
    tc = df['taille'].value_counts().sort_index()
    ax.bar(tc.index.astype(str), tc.values, color=color, edgecolor='black', alpha=0.7)
    ax.set_xlabel("Taille de l'ensemble", fontsize=12)
    ax.set_ylabel("Nombre d'observations", fontsize=12)
    ax.set_title(f'Distribution des tailles — {nom}', fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    for i, (t, c) in enumerate(tc.items()):
        ax.text(i, c + 20, str(c), ha='center', va='bottom', fontsize=9)
plt.suptitle("Distribution des tailles des ensembles de prédiction",
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/05_distribution_tailles.png', dpi=150)
plt.close()

# Figure 6 : Delta couverture PAS - Standard par espèce selon rareté
cpe_compare = cpe_std[['couverture', 'n_observations']].join(
    cpe_pas[['couverture']], rsuffix='_pas', how='inner'
)
cpe_compare['delta'] = cpe_compare['couverture_pas'] - cpe_compare['couverture']

plt.figure(figsize=(10, 6))
colors_delta = np.where(cpe_compare['delta'] > 0, 'green', 'red')
plt.scatter(cpe_compare['n_observations'], cpe_compare['delta'],
            alpha=0.5, c=colors_delta, edgecolors='white', s=50)
plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
plt.xscale('log')
plt.xlabel("Nombre d'observations par espèce (log)", fontsize=12)
plt.ylabel('Δ Couverture (PAS − Standard CP)', fontsize=12)
plt.title("Gain de couverture de PAS sur Standard CP\n(vert = PAS meilleur, rouge = Standard meilleur)",
          fontsize=13)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/06_delta_couverture_pas_vs_standard.png', dpi=150)
plt.close()

# ============================================================
# 9. SAUVEGARDE DES RÉSULTATS
# ============================================================

print("\n💾 9. Sauvegarde des résultats...")

df_standard.to_csv("results/resultats_standard.csv", index=False)
df_pas.to_csv("results/resultats_pas.csv",           index=False)
cpe_std.to_csv("results/coverage_par_espece_standard.csv", index=True)
cpe_pas.to_csv("results/coverage_par_espece_pas.csv",      index=True)

metrics_df = pd.DataFrame([
    {
        'modele':              'Standard CP',
        'alpha':               alpha,
        'couverture_cible':    0.95,
        'seuil_score':         seuil_score,
        'q_hat':               q_hat,
        'n_calibration':       n_calib,
        'n_test':              len(df_standard),
        'coverage_marginale':  m_std['coverage_marginale'],
        'coverage_macro':      m_std['coverage_macro'],
        'taille_moyenne':      m_std['taille_moyenne'],
        'taille_mediane':      m_std['taille_mediane'],
        'taille_std':          m_std['taille_std'],
        'pct_singleton':       m_std['pct_singleton'],
        'coverage_rares':      m_std['coverage_rares'],
        'coverage_communes':   m_std['coverage_communes'],
        'nb_especes_rares':    m_std['nb_especes_rares'],
        'nb_especes_communes': m_std['nb_especes_communes'],
    },
    {
        'modele':              'PAS',
        'alpha':               alpha,
        'couverture_cible':    0.95,
        'seuil_score':         np.nan,  # seuil par espèce
        'q_hat':               np.nan,
        'n_calibration':       n_calib,
        'n_test':              len(df_pas),
        'coverage_marginale':  m_pas['coverage_marginale'],
        'coverage_macro':      m_pas['coverage_macro'],
        'taille_moyenne':      m_pas['taille_moyenne'],
        'taille_mediane':      m_pas['taille_mediane'],
        'taille_std':          m_pas['taille_std'],
        'pct_singleton':       m_pas['pct_singleton'],
        'coverage_rares':      m_pas['coverage_rares'],
        'coverage_communes':   m_pas['coverage_communes'],
        'nb_especes_rares':    m_pas['nb_especes_rares'],
        'nb_especes_communes': m_pas['nb_especes_communes'],
    }
])
metrics_df.to_csv("results/metrics_phase2.csv", index=False)

print("   ✓ resultats_standard.csv")
print("   ✓ resultats_pas.csv")
print("   ✓ coverage_par_espece_standard.csv")
print("   ✓ coverage_par_espece_pas.csv")
print("   ✓ metrics_phase2.csv")
print("   ✓ 6 figures dans figures/")

# ============================================================
# 10. RÉSUMÉ FINAL
# ============================================================

print("\n" + "=" * 70)
print("RÉSUMÉ FINAL - PHASE 2")
print("=" * 70)

for m, nom in [(m_std, 'Standard CP'), (m_pas, 'PAS')]:
    obj_marg = '✅' if m['coverage_marginale'] >= 0.95 else '❌'
    obj_mac  = '✅' if m['coverage_macro']     >= 0.95 else '❌'
    print(f"""
[{nom}]
   Couverture marginale : {m['coverage_marginale']:.3f} ({m['coverage_marginale']*100:.1f}%)  {obj_marg}
   Couverture macro     : {m['coverage_macro']:.3f} ({m['coverage_macro']*100:.1f}%)  {obj_mac}
   Couverture rares     : {m['coverage_rares']:.3f} ({m['coverage_rares']*100:.1f}%)
   Taille moyenne       : {m['taille_moyenne']:.2f} espèces
   Singletons           : {m['pct_singleton']:.1f}%""")

delta_rares = m_pas['coverage_rares'] - m_std['coverage_rares']
print(f"""
💡 Gain de PAS sur Standard CP pour les espèces rares : {delta_rares:+.3f} ({delta_rares*100:+.1f} pts)

📌 Conclusion :
   La méthode standard garantit la couverture marginale mais pénalise les
   espèces rares. PAS adapte le seuil par espèce (min. {MIN_OBS_PAS} obs) et améliore
   la couverture macro, en particulier sur les espèces peu représentées.
""")

print("\n✅ PHASE 2 TERMINÉE AVEC SUCCÈS !")
print("=" * 70)