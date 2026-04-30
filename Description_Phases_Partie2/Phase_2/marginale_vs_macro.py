"""
=============================================================================
Partie 2 — Phase 2 : Couverture Marginale vs Couverture Macro
=============================================================================
Objectif : Répondre à l'exigence du prof sur la différence de performance
           entre CP Standard et PAS, surtout sur les espèces rares.

Métriques :
    - Couverture MARGINALE : taux de succès global sur le test set
      (les espèces communes dominent ce chiffre)
    - Couverture MACRO : pour chaque espèce, calculer son taux de succès
      individuel, puis faire la MOYENNE de ces taux
      (chaque espèce compte autant, rare ou commune)

Données : calibration.csv, test.csv, ai_scores_all.csv
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================================================================
# CONFIGURATION
# ================================================================
ALPHA = 0.05
SEUIL_RARETE = 10

# ================================================================
# ÉTAPE 1 : CHARGEMENT
# ================================================================
print("=" * 65)
print("PARTIE 2 — PHASE 2 : COUVERTURE MARGINALE vs MACRO")
print("=" * 65)

calib = pd.read_csv("calibration.csv")
test = pd.read_csv("test.csv")
scores = pd.read_csv("ai_scores_all.csv")

print(f"\nCalibration : {len(calib):,} observations")
print(f"Test        : {len(test):,} observations")

# ================================================================
# ÉTAPE 2 : PRÉVALENCE ET GROUPES
# ================================================================
prevalence = calib["ground_truth_val"].value_counts()
groupes = {esp: ("Rare" if n < SEUIL_RARETE else "Commun")
           for esp, n in prevalence.items()}

# ================================================================
# ÉTAPE 3 : SCORES DE NON-CONFORMITÉ (avec correction)
# ================================================================
print(f"\nCalcul des scores de non-conformité (avec correction)...")

calib_scores = scores[scores["observation_id"].isin(calib["observation_id"])]
merged_cal = calib_scores.merge(calib, on="observation_id")

vraie_cal = merged_cal[
    merged_cal["spicies_id"] == merged_cal["ground_truth_val"]
].copy()
vraie_cal["non_conformity"] = 1 - vraie_cal["score"]

# Correction des observations manquantes
obs_trouvees = set(vraie_cal["observation_id"].unique())
obs_totales = set(calib["observation_id"].unique())
obs_manquantes = obs_totales - obs_trouvees

manquantes_df = calib[calib["observation_id"].isin(obs_manquantes)].copy()
manquantes_df["non_conformity"] = 1.0

cols = ["observation_id", "ground_truth_val", "non_conformity"]
toutes_cal = pd.concat([vraie_cal[cols], manquantes_df[cols]], ignore_index=True)
toutes_cal["groupe"] = toutes_cal["ground_truth_val"].map(groupes)

print(f"  Observations corrigées : {len(obs_manquantes):,} (S_i = 1.0)")

# ================================================================
# ÉTAPE 4 : QUANTILES
# ================================================================

# Standard : un seul quantile
scores_all = toutes_cal["non_conformity"].values
n_all = len(scores_all)
niveau_all = min(np.ceil((n_all + 1) * (1 - ALPHA)) / n_all, 1.0)
q_hat_std = np.quantile(scores_all, niveau_all, method="higher")
seuil_std = 1 - q_hat_std

# PAS : un quantile par groupe
scores_rare = toutes_cal[toutes_cal["groupe"] == "Rare"]["non_conformity"].values
scores_commun = toutes_cal[toutes_cal["groupe"] == "Commun"]["non_conformity"].values

n_r, n_c = len(scores_rare), len(scores_commun)
niv_r = min(np.ceil((n_r + 1) * (1 - ALPHA)) / n_r, 1.0)
niv_c = min(np.ceil((n_c + 1) * (1 - ALPHA)) / n_c, 1.0)

q_hat_rare = np.quantile(scores_rare, niv_r, method="higher")
q_hat_commun = np.quantile(scores_commun, niv_c, method="higher")
seuil_rare = 1 - q_hat_rare
seuil_commun = 1 - q_hat_commun

print(f"\n  Standard : q̂ = {q_hat_std:.6f}")
print(f"  PAS Rare : q̂ = {q_hat_rare:.6f}, PAS Commun : q̂ = {q_hat_commun:.6f}")

# ================================================================
# ÉTAPE 5 : ENSEMBLES DE PRÉDICTION
# ================================================================
print(f"\nConstruction des ensembles de prédiction...")

test_scores = scores[scores["observation_id"].isin(test["observation_id"])]
merged_test = test_scores.merge(test, on="observation_id")

resultats_std = []
resultats_pas = []

obs_ids = merged_test["observation_id"].unique()
for i, obs_id in enumerate(obs_ids):
    if (i + 1) % 2000 == 0:
        print(f"  {i + 1:,} / {len(obs_ids):,}")

    lignes = merged_test[merged_test["observation_id"] == obs_id]
    vrai = lignes["ground_truth_val"].iloc[0]
    groupe_vrai = groupes.get(vrai, "Rare")

    # Standard
    gardees_std = lignes[lignes["score"] >= seuil_std]["spicies_id"].tolist()
    if len(gardees_std) == 0:
        gardees_std = [lignes.loc[lignes["score"].idxmax(), "spicies_id"]]

    resultats_std.append({
        "observation_id": obs_id,
        "vraie_espece": vrai,
        "groupe": groupe_vrai,
        "taille": len(gardees_std),
        "couvert": vrai in gardees_std
    })

    # PAS
    gardees_pas = []
    for _, row in lignes.iterrows():
        g = groupes.get(row["spicies_id"], "Rare")
        s = seuil_rare if g == "Rare" else seuil_commun
        if row["score"] >= s:
            gardees_pas.append(row["spicies_id"])
    if len(gardees_pas) == 0:
        gardees_pas = [lignes.loc[lignes["score"].idxmax(), "spicies_id"]]

    resultats_pas.append({
        "observation_id": obs_id,
        "vraie_espece": vrai,
        "groupe": groupe_vrai,
        "taille": len(gardees_pas),
        "couvert": vrai in gardees_pas
    })

df_std = pd.DataFrame(resultats_std)
df_pas = pd.DataFrame(resultats_pas)

# ================================================================
# ÉTAPE 6 : CALCUL DES COUVERTURES PAR ESPÈCE
# ================================================================
print(f"\n{'=' * 65}")
print("COUVERTURE PAR ESPÈCE")
print(f"{'=' * 65}")

# Couverture par espèce pour Standard
couv_espece_std = df_std.groupby("vraie_espece").agg(
    couverture=("couvert", "mean"),
    n_obs=("couvert", "count"),
    groupe=("groupe", "first")
).reset_index()

# Couverture par espèce pour PAS
couv_espece_pas = df_pas.groupby("vraie_espece").agg(
    couverture=("couvert", "mean"),
    n_obs=("couvert", "count"),
    groupe=("groupe", "first")
).reset_index()

# Fusionner pour comparaison
couv_compare = couv_espece_std[["vraie_espece", "couverture", "n_obs", "groupe"]].rename(
    columns={"couverture": "couv_std"}
).merge(
    couv_espece_pas[["vraie_espece", "couverture"]].rename(
        columns={"couverture": "couv_pas"}
    ),
    on="vraie_espece"
)
couv_compare["diff"] = couv_compare["couv_pas"] - couv_compare["couv_std"]

n_especes_test = len(couv_compare)
print(f"  Espèces dans le test : {n_especes_test:,}")

# ================================================================
# ÉTAPE 7 : MÉTRIQUES MARGINALE vs MACRO
# ================================================================
print(f"\n{'=' * 65}")
print("RÉSULTATS : MARGINALE vs MACRO")
print(f"{'=' * 65}")

# Marginale
couv_marg_std = df_std["couvert"].mean()
couv_marg_pas = df_pas["couvert"].mean()

# Macro (moyenne des couvertures par espèce)
couv_macro_std = couv_espece_std["couverture"].mean()
couv_macro_pas = couv_espece_pas["couverture"].mean()

# Par groupe
for groupe in ["Rare", "Commun"]:
    sub_std = couv_espece_std[couv_espece_std["groupe"] == groupe]
    sub_pas = couv_espece_pas[couv_espece_pas["groupe"] == groupe]
    macro_std = sub_std["couverture"].mean()
    macro_pas = sub_pas["couverture"].mean()
    marg_std = df_std[df_std["groupe"] == groupe]["couvert"].mean()
    marg_pas = df_pas[df_pas["groupe"] == groupe]["couvert"].mean()
    print(f"\n  Groupe {groupe} ({len(sub_std):,} espèces) :")
    print(f"    Marginale : Standard = {marg_std:.4f}, PAS = {marg_pas:.4f}")
    print(f"    Macro     : Standard = {macro_std:.4f}, PAS = {macro_pas:.4f}")

print(f"\n  {'='*55}")
print(f"  {'Métrique':<30} {'Standard':>10} {'PAS':>10} {'Δ':>10}")
print(f"  {'-'*55}")
print(f"  {'Couverture MARGINALE':<30} {couv_marg_std:>10.4f} {couv_marg_pas:>10.4f} "
      f"{couv_marg_pas - couv_marg_std:>+10.4f}")
print(f"  {'Couverture MACRO':<30} {couv_macro_std:>10.4f} {couv_macro_pas:>10.4f} "
      f"{couv_macro_pas - couv_macro_std:>+10.4f}")
print(f"  {'Taille moyenne':<30} {df_std['taille'].mean():>10.2f} "
      f"{df_pas['taille'].mean():>10.2f} "
      f"{df_pas['taille'].mean() - df_std['taille'].mean():>+10.2f}")

# Espèces où PAS améliore / dégrade
n_ameliore = (couv_compare["diff"] > 0).sum()
n_degrade = (couv_compare["diff"] < 0).sum()
n_egal = (couv_compare["diff"] == 0).sum()

print(f"\n  Impact de PAS sur les espèces individuelles :")
print(f"    Améliorées : {n_ameliore:,} espèces")
print(f"    Dégradées  : {n_degrade:,} espèces")
print(f"    Inchangées : {n_egal:,} espèces")

# ================================================================
# ÉTAPE 8 : STATISTIQUES DES COUVERTURES PAR ESPÈCE
# ================================================================
print(f"\n{'=' * 65}")
print("DISTRIBUTION DES COUVERTURES PAR ESPÈCE")
print(f"{'=' * 65}")

for nom, couv_df in [("Standard", couv_espece_std), ("PAS", couv_espece_pas)]:
    print(f"\n  --- {nom} ---")
    print(f"    Moyenne (= couv. macro) : {couv_df['couverture'].mean():.4f}")
    print(f"    Médiane                 : {couv_df['couverture'].median():.4f}")
    print(f"    Écart-type              : {couv_df['couverture'].std():.4f}")
    print(f"    Min                     : {couv_df['couverture'].min():.4f}")
    print(f"    % espèces à 100%        : "
          f"{100 * (couv_df['couverture'] == 1.0).mean():.1f}%")
    print(f"    % espèces à 0%          : "
          f"{100 * (couv_df['couverture'] == 0.0).mean():.1f}%")
    print(f"    % espèces ≥ 95%         : "
          f"{100 * (couv_df['couverture'] >= 0.95).mean():.1f}%")
    print(f"    % espèces < 80%         : "
          f"{100 * (couv_df['couverture'] < 0.80).mean():.1f}%")

# Sauvegarder les couvertures par espèce
couv_compare.to_csv("couverture_par_espece.csv", index=False)
print(f"\n✓ couverture_par_espece.csv")

# ================================================================
# ÉTAPE 9 : VISUALISATIONS
# ================================================================
print(f"\n{'=' * 65}")
print("GÉNÉRATION DES FIGURES")
print(f"{'=' * 65}")

# --- Figure 1 : Boxplot des couvertures par espèce ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Toutes espèces
data_all = [couv_espece_std["couverture"].values,
            couv_espece_pas["couverture"].values]
bp1 = axes[0].boxplot(data_all, labels=["Standard", "PAS"],
                       patch_artist=True, showmeans=True,
                       meanprops={"marker": "D", "markerfacecolor": "red",
                                  "markersize": 8})
bp1["boxes"][0].set_facecolor("lightcoral")
bp1["boxes"][1].set_facecolor("steelblue")
for box in bp1["boxes"]:
    box.set_alpha(0.7)
axes[0].axhline(y=1 - ALPHA, color="green", linestyle="--", linewidth=1.5,
                label=f"Cible = {1 - ALPHA:.0%}")
axes[0].set_ylabel("Couverture par espèce")
axes[0].set_title(f"Toutes les espèces ({n_especes_test:,})\n"
                  f"◆ = moyenne (couv. macro)")
axes[0].legend()
axes[0].set_ylim(-0.05, 1.1)

# Par groupe (Rares seulement)
rares_std = couv_espece_std[couv_espece_std["groupe"] == "Rare"]["couverture"].values
rares_pas = couv_espece_pas[couv_espece_pas["groupe"] == "Rare"]["couverture"].values
n_rares = len(rares_std)

data_rares = [rares_std, rares_pas]
bp2 = axes[1].boxplot(data_rares, labels=["Standard", "PAS"],
                       patch_artist=True, showmeans=True,
                       meanprops={"marker": "D", "markerfacecolor": "red",
                                  "markersize": 8})
bp2["boxes"][0].set_facecolor("lightcoral")
bp2["boxes"][1].set_facecolor("steelblue")
for box in bp2["boxes"]:
    box.set_alpha(0.7)
axes[1].axhline(y=1 - ALPHA, color="green", linestyle="--", linewidth=1.5,
                label=f"Cible = {1 - ALPHA:.0%}")
axes[1].set_ylabel("Couverture par espèce")
axes[1].set_title(f"Espèces RARES uniquement ({n_rares:,})\n"
                  f"◆ = moyenne (couv. macro)")
axes[1].legend()
axes[1].set_ylim(-0.05, 1.1)

fig.suptitle("Distribution des couvertures par espèce : Standard vs PAS",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("fig_boxplot_couverture_par_espece.png", dpi=150,
            bbox_inches="tight")
plt.close()
print(f"✓ fig_boxplot_couverture_par_espece.png")

# --- Figure 2 : Histogramme des couvertures par espèce ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

bins_couv = np.arange(0, 1.1, 0.1)

axes[0].hist(couv_espece_std["couverture"], bins=bins_couv, alpha=0.6,
             color="lightcoral", edgecolor="black", label="Standard")
axes[0].hist(couv_espece_pas["couverture"], bins=bins_couv, alpha=0.6,
             color="steelblue", edgecolor="black", label="PAS")
axes[0].axvline(x=1 - ALPHA, color="green", linestyle="--", linewidth=1.5,
                label=f"Cible = {1 - ALPHA:.0%}")
axes[0].set_xlabel("Couverture de l'espèce")
axes[0].set_ylabel("Nombre d'espèces")
axes[0].set_title(f"Toutes les espèces ({n_especes_test:,})")
axes[0].legend(fontsize=9)

# Zoom sur les espèces rares
rares_std_df = couv_espece_std[couv_espece_std["groupe"] == "Rare"]
rares_pas_df = couv_espece_pas[couv_espece_pas["groupe"] == "Rare"]

axes[1].hist(rares_std_df["couverture"], bins=bins_couv, alpha=0.6,
             color="lightcoral", edgecolor="black", label="Standard")
axes[1].hist(rares_pas_df["couverture"], bins=bins_couv, alpha=0.6,
             color="steelblue", edgecolor="black", label="PAS")
axes[1].axvline(x=1 - ALPHA, color="green", linestyle="--", linewidth=1.5,
                label=f"Cible = {1 - ALPHA:.0%}")
axes[1].set_xlabel("Couverture de l'espèce")
axes[1].set_ylabel("Nombre d'espèces")
axes[1].set_title(f"Espèces RARES ({n_rares:,})")
axes[1].legend(fontsize=9)

fig.suptitle("Histogramme des couvertures par espèce",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("fig_histogramme_couverture_par_espece.png", dpi=150,
            bbox_inches="tight")
plt.close()
print(f"✓ fig_histogramme_couverture_par_espece.png")

# --- Figure 3 : Couverture vs prévalence (scatter) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for i, (nom, couv_df) in enumerate([("Standard", couv_espece_std),
                                     ("PAS", couv_espece_pas)]):
    color = "lightcoral" if nom == "Standard" else "steelblue"

    # Prévalence de chaque espèce dans le test
    axes[i].scatter(couv_df["n_obs"], couv_df["couverture"],
                    alpha=0.3, s=15, color=color, edgecolors="none")
    axes[i].axhline(y=1 - ALPHA, color="green", linestyle="--", linewidth=1.5,
                    label=f"Cible = {1 - ALPHA:.0%}")
    axes[i].axvline(x=SEUIL_RARETE, color="red", linestyle=":",
                    linewidth=1, label=f"Seuil rareté = {SEUIL_RARETE}")
    axes[i].set_xlabel("Nombre d'observations (test)")
    axes[i].set_ylabel("Couverture de l'espèce")
    axes[i].set_title(f"{nom}")
    axes[i].legend(fontsize=8)
    axes[i].set_ylim(-0.05, 1.1)

fig.suptitle("Couverture par espèce en fonction de la prévalence",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("fig_couverture_vs_prevalence.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ fig_couverture_vs_prevalence.png")

# --- Figure 4 : Résumé marginale vs macro ---
fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(2)
w = 0.3
bars1 = ax.bar(x - w/2, [couv_marg_std, couv_macro_std], w,
               label="Standard", color="lightcoral", edgecolor="black")
bars2 = ax.bar(x + w/2, [couv_marg_pas, couv_macro_pas], w,
               label="PAS", color="steelblue", edgecolor="black")
ax.axhline(y=1 - ALPHA, color="green", linestyle="--", linewidth=1.5,
           label=f"Cible = {1 - ALPHA:.0%}")
ax.set_ylabel("Couverture")
ax.set_title("Couverture marginale vs macro : Standard vs PAS",
             fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(["Marginale\n(moyenne sur les obs)",
                     "Macro\n(moyenne sur les espèces)"])
ax.legend()
ax.set_ylim(0.90, 1.0)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f"{bar.get_height():.4f}", ha="center", fontsize=10)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f"{bar.get_height():.4f}", ha="center", fontsize=10)

plt.tight_layout()
plt.savefig("fig_marginale_vs_macro.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ fig_marginale_vs_macro.png")

# ================================================================
# CONCLUSION
# ================================================================
print(f"\n{'=' * 65}")
print("CONCLUSION")
print(f"{'=' * 65}")

print(f"""
  COUVERTURE MARGINALE (moyenne sur les observations) :
    Standard = {couv_marg_std:.4f}
    PAS      = {couv_marg_pas:.4f}
    → Les deux méthodes atteignent ~95% (comme prédit par la théorie)

  COUVERTURE MACRO (moyenne sur les espèces) :
    Standard = {couv_macro_std:.4f}
    PAS      = {couv_macro_pas:.4f}
    → PAS {'améliore' if couv_macro_pas > couv_macro_std else 'est similaire à'} la couverture macro

  La couverture marginale est dominée par les espèces communes
  (qui ont beaucoup d'observations). La couverture macro donne
  un poids égal à chaque espèce, révélant les faiblesses sur
  les espèces rares.
""")

print("✅ Terminé")
