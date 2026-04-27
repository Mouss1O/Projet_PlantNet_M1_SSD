"""
=============================================================================
Partie 2 — Phase 3 : CP Conditionnelle + Comparaison des 3 méthodes
=============================================================================
Objectif :
    1. Corriger le biais des observations manquantes (déjà fait dans PAS)
    2. Implémenter la CP conditionnelle (un seuil par espèce)
    3. Montrer que la CP conditionnelle produit des ensembles ÉNORMES
    4. Comparer les 3 méthodes : Standard, PAS, Conditionnelle

Demandé par : Joseph Salmon (mail du 21/04/2026)
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

# Fichiers d'entrée (même dossier)
PATH_CAL = "calibration.csv"
PATH_TEST = "test.csv"
PATH_SCORES = "ai_scores_all.csv"

# ================================================================
# ÉTAPE 1 : CHARGEMENT
# ================================================================
print("=" * 65)
print("PARTIE 2 — PHASE 3 : CP CONDITIONNELLE")
print("=" * 65)

calib = pd.read_csv(PATH_CAL)
test = pd.read_csv(PATH_TEST)
scores = pd.read_csv(PATH_SCORES)

print(f"Calibration : {len(calib):,} observations")
print(f"Test        : {len(test):,} observations")
print(f"Scores      : {len(scores):,} lignes")

# ================================================================
# ÉTAPE 2 : PRÉVALENCE ET GROUPES
# ================================================================
prevalence = calib["ground_truth_val"].value_counts()
groupes = {esp: ("Rare" if n < SEUIL_RARETE else "Commun")
           for esp, n in prevalence.items()}

n_rares = sum(1 for g in groupes.values() if g == "Rare")
n_communs = sum(1 for g in groupes.values() if g == "Commun")
print(f"\nEspèces : {len(groupes):,} ({n_rares:,} rares, {n_communs:,} communes)")

# ================================================================
# ÉTAPE 3 : CALCUL DES SCORES DE NON-CONFORMITÉ (avec correction)
# ================================================================
print(f"\n{'=' * 65}")
print("CALCUL DES SCORES DE NON-CONFORMITÉ (avec correction)")
print(f"{'=' * 65}")

# Filtrer les scores pour la calibration
calib_scores = scores[scores["observation_id"].isin(calib["observation_id"])]
merged_cal = calib_scores.merge(calib, on="observation_id")

# Score pour la vraie espèce
vraie_cal = merged_cal[
    merged_cal["spicies_id"] == merged_cal["ground_truth_val"]
].copy()
vraie_cal["non_conformity"] = 1 - vraie_cal["score"]

# Correction : observations manquantes → s_i = 1.0
obs_trouvees = set(vraie_cal["observation_id"].unique())
obs_totales = set(calib["observation_id"].unique())
obs_manquantes = obs_totales - obs_trouvees

print(f"  Observations avec vraie espèce trouvée  : {len(obs_trouvees):,}")
print(f"  Observations manquantes (s_i = 1.0)     : {len(obs_manquantes):,}")

manquantes_df = calib[calib["observation_id"].isin(obs_manquantes)].copy()
manquantes_df["non_conformity"] = 1.0
manquantes_df["spicies_id"] = manquantes_df["ground_truth_val"]
manquantes_df["score"] = 0.0

cols = ["observation_id", "ground_truth_val", "non_conformity"]
toutes_cal = pd.concat([vraie_cal[cols], manquantes_df[cols]], ignore_index=True)

# ================================================================
# ÉTAPE 4 : PRÉPARER LE TEST SET
# ================================================================
test_scores = scores[scores["observation_id"].isin(test["observation_id"])]
merged_test = test_scores.merge(test, on="observation_id")

# ================================================================
# MÉTHODE 1 : CP STANDARD (un seul quantile global, AVEC correction)
# ================================================================
print(f"\n{'=' * 65}")
print("MÉTHODE 1 : CP STANDARD (corrigée)")
print(f"{'=' * 65}")

scores_all = toutes_cal["non_conformity"].values
n = len(scores_all)
niveau = min(np.ceil((n + 1) * (1 - ALPHA)) / n, 1.0)
q_hat_global = np.quantile(scores_all, niveau, method="higher")
seuil_global = 1 - q_hat_global

print(f"  n = {n:,}, q̂ = {q_hat_global:.6f}, seuil = {seuil_global:.6f}")

resultats_std = []
for obs_id in merged_test["observation_id"].unique():
    lignes = merged_test[merged_test["observation_id"] == obs_id]
    gardees = lignes[lignes["score"] >= seuil_global]["spicies_id"].tolist()
    if len(gardees) == 0:
        gardees = [lignes.loc[lignes["score"].idxmax(), "spicies_id"]]
    vrai = lignes["ground_truth_val"].iloc[0]
    resultats_std.append({
        "observation_id": obs_id,
        "vraie_espece": vrai,
        "groupe": groupes.get(vrai, "Rare"),
        "taille": len(gardees),
        "couvert": vrai in gardees
    })

df_std = pd.DataFrame(resultats_std)

# ================================================================
# MÉTHODE 2 : PAS (un quantile par groupe, AVEC correction)
# ================================================================
print(f"\n{'=' * 65}")
print("MÉTHODE 2 : PAS (corrigée)")
print(f"{'=' * 65}")

toutes_cal["groupe"] = toutes_cal["ground_truth_val"].map(groupes)

scores_rare = toutes_cal[toutes_cal["groupe"] == "Rare"]["non_conformity"].values
scores_commun = toutes_cal[toutes_cal["groupe"] == "Commun"]["non_conformity"].values

n_r, n_c = len(scores_rare), len(scores_commun)
niv_r = min(np.ceil((n_r + 1) * (1 - ALPHA)) / n_r, 1.0)
niv_c = min(np.ceil((n_c + 1) * (1 - ALPHA)) / n_c, 1.0)

q_hat_rare = np.quantile(scores_rare, niv_r, method="higher")
q_hat_commun = np.quantile(scores_commun, niv_c, method="higher")

seuil_rare = 1 - q_hat_rare
seuil_commun = 1 - q_hat_commun

print(f"  Rares   : n = {n_r:,}, q̂ = {q_hat_rare:.6f}, seuil = {seuil_rare:.6f}")
print(f"  Communes: n = {n_c:,}, q̂ = {q_hat_commun:.6f}, seuil = {seuil_commun:.6f}")

resultats_pas = []
for obs_id in merged_test["observation_id"].unique():
    lignes = merged_test[merged_test["observation_id"] == obs_id]
    gardees = []
    for _, row in lignes.iterrows():
        g = groupes.get(row["spicies_id"], "Rare")
        s = seuil_rare if g == "Rare" else seuil_commun
        if row["score"] >= s:
            gardees.append(row["spicies_id"])
    if len(gardees) == 0:
        gardees = [lignes.loc[lignes["score"].idxmax(), "spicies_id"]]
    vrai = lignes["ground_truth_val"].iloc[0]
    resultats_pas.append({
        "observation_id": obs_id,
        "vraie_espece": vrai,
        "groupe": groupes.get(vrai, "Rare"),
        "taille": len(gardees),
        "couvert": vrai in gardees
    })

df_pas = pd.DataFrame(resultats_pas)

# ================================================================
# MÉTHODE 3 : CP CONDITIONNELLE (un quantile par espèce)
# ================================================================
print(f"\n{'=' * 65}")
print("MÉTHODE 3 : CP CONDITIONNELLE (un quantile par espèce)")
print(f"{'=' * 65}")

# Calculer un quantile par espèce
q_hat_par_espece = {}
stats_par_espece = {}

for espece in toutes_cal["ground_truth_val"].unique():
    scores_k = toutes_cal[
        toutes_cal["ground_truth_val"] == espece
    ]["non_conformity"].values

    n_k = len(scores_k)
    if n_k == 0:
        q_hat_par_espece[espece] = 1.0
        stats_par_espece[espece] = 0
        continue

    niveau_k = min(np.ceil((n_k + 1) * (1 - ALPHA)) / n_k, 1.0)
    q_k = np.quantile(scores_k, niveau_k, method="higher")
    q_hat_par_espece[espece] = q_k
    stats_par_espece[espece] = n_k

# Statistiques sur les quantiles par espèce
q_values = list(q_hat_par_espece.values())
n_values = list(stats_par_espece.values())

n_q_egal_1 = sum(1 for q in q_values if q >= 0.999)
print(f"  Espèces avec quantile calculé : {len(q_hat_par_espece):,}")
print(f"  Espèces avec q̂ = 1.0          : {n_q_egal_1:,} "
      f"({100*n_q_egal_1/len(q_hat_par_espece):.1f}%)")
print(f"  Médiane des n_k               : {np.median(n_values):.0f}")
print(f"  Moyenne des q̂_k               : {np.mean(q_values):.4f}")

# Construction des ensembles conditionnels
resultats_cond = []

for obs_id in merged_test["observation_id"].unique():
    lignes = merged_test[merged_test["observation_id"] == obs_id]
    gardees = []

    for _, row in lignes.iterrows():
        espece_candidate = row["spicies_id"]
        # Seuil spécifique à cette espèce candidate
        q_k = q_hat_par_espece.get(espece_candidate, 1.0)
        seuil_k = 1 - q_k

        if row["score"] >= seuil_k:
            gardees.append(espece_candidate)

    if len(gardees) == 0:
        gardees = [lignes.loc[lignes["score"].idxmax(), "spicies_id"]]

    vrai = lignes["ground_truth_val"].iloc[0]
    resultats_cond.append({
        "observation_id": obs_id,
        "vraie_espece": vrai,
        "groupe": groupes.get(vrai, "Rare"),
        "taille": len(gardees),
        "couvert": vrai in gardees
    })

df_cond = pd.DataFrame(resultats_cond)

# ================================================================
# ÉTAPE 5 : CALCUL DES MÉTRIQUES POUR LES 3 MÉTHODES
# ================================================================
def calculer_metriques(df, nom):
    """Calcule couverture marginale, macro, et taille moyenne."""
    couv_marg = df["couvert"].mean()
    taille_moy = df["taille"].mean()
    taille_med = df["taille"].median()

    # Couverture macro (moyenne des couvertures par espèce)
    couv_par_espece = df.groupby("vraie_espece")["couvert"].mean()
    couv_macro = couv_par_espece.mean()

    # Par groupe
    couv_marg_rare = df[df["groupe"] == "Rare"]["couvert"].mean()
    couv_marg_comm = df[df["groupe"] == "Commun"]["couvert"].mean()

    couv_macro_rare = df[df["groupe"] == "Rare"].groupby(
        "vraie_espece")["couvert"].mean().mean()
    couv_macro_comm = df[df["groupe"] == "Commun"].groupby(
        "vraie_espece")["couvert"].mean().mean()

    taille_rare = df[df["groupe"] == "Rare"]["taille"].mean()
    taille_comm = df[df["groupe"] == "Commun"]["taille"].mean()

    return {
        "Méthode": nom,
        "Couv. marginale": couv_marg,
        "Couv. macro": couv_macro,
        "Couv. marg. Rares": couv_marg_rare,
        "Couv. marg. Communes": couv_marg_comm,
        "Couv. macro Rares": couv_macro_rare,
        "Couv. macro Communes": couv_macro_comm,
        "Taille moyenne": taille_moy,
        "Taille médiane": taille_med,
        "Taille Rares": taille_rare,
        "Taille Communes": taille_comm,
    }

m_std = calculer_metriques(df_std, "Standard")
m_pas = calculer_metriques(df_pas, "PAS")
m_cond = calculer_metriques(df_cond, "Conditionnelle")

# ================================================================
# ÉTAPE 6 : AFFICHAGE DU TABLEAU COMPARATIF
# ================================================================
print(f"\n{'=' * 65}")
print("COMPARAISON DES 3 MÉTHODES")
print(f"{'=' * 65}")

header = f"  {'Métrique':<25} {'Standard':>10} {'PAS':>10} {'Cond.':>10}"
print(header)
print(f"  {'-' * 55}")

for key in ["Couv. marginale", "Couv. macro",
            "Couv. marg. Rares", "Couv. marg. Communes",
            "Couv. macro Rares", "Couv. macro Communes",
            "Taille moyenne", "Taille médiane",
            "Taille Rares", "Taille Communes"]:
    v1 = m_std[key]
    v2 = m_pas[key]
    v3 = m_cond[key]
    if "Taille" in key:
        print(f"  {key:<25} {v1:>10.2f} {v2:>10.2f} {v3:>10.2f}")
    else:
        print(f"  {key:<25} {v1:>10.4f} {v2:>10.4f} {v3:>10.4f}")

# Sauvegarder le tableau
comparaison = pd.DataFrame([m_std, m_pas, m_cond])
comparaison.to_csv("comparaison_3_methodes.csv", index=False)
print(f"\n✓ comparaison_3_methodes.csv")

# ================================================================
# ÉTAPE 7 : VISUALISATIONS
# ================================================================
print(f"\n{'=' * 65}")
print("GÉNÉRATION DES FIGURES")
print(f"{'=' * 65}")

# --- Figure 1 : Couverture marginale et macro pour les 3 méthodes ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Couvertures
methodes = ["Standard\n(corrigée)", "PAS", "Conditionnelle"]
couv_marg = [m_std["Couv. marginale"], m_pas["Couv. marginale"],
             m_cond["Couv. marginale"]]
couv_macro = [m_std["Couv. macro"], m_pas["Couv. macro"],
              m_cond["Couv. macro"]]

x = np.arange(3)
w = 0.35
bars1 = axes[0].bar(x - w/2, couv_marg, w, label="Marginale",
                     color="steelblue", edgecolor="black")
bars2 = axes[0].bar(x + w/2, couv_macro, w, label="Macro",
                     color="tomato", edgecolor="black")
axes[0].axhline(y=1-ALPHA, color="green", linestyle="--", linewidth=1.5,
                label=f"Cible = {1-ALPHA:.0%}")
axes[0].set_ylabel("Couverture")
axes[0].set_title("Couverture marginale vs macro")
axes[0].set_xticks(x)
axes[0].set_xticklabels(methodes, fontsize=9)
axes[0].legend(fontsize=8)
axes[0].set_ylim(0.80, 1.02)

for bar in bars1:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{bar.get_height():.3f}", ha="center", fontsize=8)
for bar in bars2:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{bar.get_height():.3f}", ha="center", fontsize=8)

# --- Figure 2 : Distribution des tailles (3 méthodes superposées) ---
taille_max_plot = min(int(df_cond["taille"].quantile(0.95)) + 5, 50)
bins = range(1, taille_max_plot + 1)

axes[1].hist(df_std["taille"].clip(upper=taille_max_plot), bins=bins,
             alpha=0.5, label=f"Standard (moy={m_std['Taille moyenne']:.1f})",
             color="steelblue", edgecolor="black", align="left")
axes[1].hist(df_pas["taille"].clip(upper=taille_max_plot), bins=bins,
             alpha=0.5, label=f"PAS (moy={m_pas['Taille moyenne']:.1f})",
             color="orange", edgecolor="black", align="left")
axes[1].hist(df_cond["taille"].clip(upper=taille_max_plot), bins=bins,
             alpha=0.5, label=f"Cond. (moy={m_cond['Taille moyenne']:.1f})",
             color="tomato", edgecolor="black", align="left")
axes[1].set_xlabel("Taille de l'ensemble C(x)")
axes[1].set_ylabel("Nombre d'observations")
axes[1].set_title("Distribution des tailles des ensembles")
axes[1].legend(fontsize=8)

# --- Figure 3 : Boxplot des tailles par méthode ---
data_box = [df_std["taille"].values, df_pas["taille"].values,
            df_cond["taille"].values]
bp = axes[2].boxplot(data_box, labels=["Standard", "PAS", "Conditionnelle"],
                     patch_artist=True, showfliers=False)
colors_box = ["steelblue", "orange", "tomato"]
for patch, color in zip(bp["boxes"], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[2].set_ylabel("Taille de l'ensemble C(x)")
axes[2].set_title("Taille des ensembles (sans outliers)")

fig.suptitle("Comparaison des 3 méthodes de Conformal Prediction",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("fig_comparaison_3_methodes.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ fig_comparaison_3_methodes.png")

# --- Figure 4 : Couverture par groupe pour les 3 méthodes ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for i, (type_couv, titre) in enumerate([
    ("marg", "Couverture MARGINALE par groupe"),
    ("macro", "Couverture MACRO par groupe")
]):
    groupes_labels = ["Rares", "Communes"]
    key_r = f"Couv. {type_couv}. Rares" if type_couv == "marg" else f"Couv. {type_couv} Rares"
    key_c = f"Couv. {type_couv}. Communes" if type_couv == "marg" else f"Couv. {type_couv} Communes"

    vals_std = [m_std[key_r], m_std[key_c]]
    vals_pas = [m_pas[key_r], m_pas[key_c]]
    vals_cond = [m_cond[key_r], m_cond[key_c]]

    x = np.arange(2)
    w = 0.25
    axes[i].bar(x - w, vals_std, w, label="Standard", color="steelblue",
                edgecolor="black")
    axes[i].bar(x, vals_pas, w, label="PAS", color="orange",
                edgecolor="black")
    axes[i].bar(x + w, vals_cond, w, label="Conditionnelle", color="tomato",
                edgecolor="black")
    axes[i].axhline(y=1-ALPHA, color="green", linestyle="--", linewidth=1.5,
                    label=f"Cible = {1-ALPHA:.0%}")
    axes[i].set_ylabel("Couverture")
    axes[i].set_title(titre)
    axes[i].set_xticks(x)
    axes[i].set_xticklabels(groupes_labels)
    axes[i].legend(fontsize=8)
    axes[i].set_ylim(0.80, 1.02)

plt.tight_layout()
plt.savefig("fig_couverture_par_groupe_3_methodes.png", dpi=150,
            bbox_inches="tight")
plt.close()
print(f"✓ fig_couverture_par_groupe_3_methodes.png")

# ================================================================
# CONCLUSION
# ================================================================
print(f"\n{'=' * 65}")
print("CONCLUSION")
print(f"{'=' * 65}")

print(f"""
Résultats clés :

1. COUVERTURE MARGINALE : les 3 méthodes sont proches de 95%.
   Standard = {m_std['Couv. marginale']:.4f}, PAS = {m_pas['Couv. marginale']:.4f}, 
   Cond. = {m_cond['Couv. marginale']:.4f}

2. COUVERTURE MACRO : PAS améliore la couverture macro par rapport
   au Standard. La Conditionnelle a la meilleure couverture macro
   mais au prix d'ensembles énormes.
   Standard = {m_std['Couv. macro']:.4f}, PAS = {m_pas['Couv. macro']:.4f}, 
   Cond. = {m_cond['Couv. macro']:.4f}

3. TAILLE DES ENSEMBLES : c'est le problème de la Conditionnelle.
   Standard = {m_std['Taille moyenne']:.1f}, PAS = {m_pas['Taille moyenne']:.1f}, 
   Cond. = {m_cond['Taille moyenne']:.1f}
   → La Conditionnelle produit des ensembles {m_cond['Taille moyenne']/m_std['Taille moyenne']:.0f}x 
     plus grands que le Standard.
   → Cela rend les prédictions INUTILES pour l'utilisateur.

4. PAS est le MEILLEUR COMPROMIS : il améliore la couverture macro
   sans exploser la taille des ensembles.
""")

print(f"✅ Terminé")
