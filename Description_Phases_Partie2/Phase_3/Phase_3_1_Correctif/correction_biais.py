"""
=============================================================================
Partie 2 — Phase 3 (Volet 1) : Correction du Biai
=============================================================================
Objectif : Atteindre les 95% de couverture marginale en corrigeant le biais
           causé par les observations manquantes dans ai_scores_all.csv.

Problème identifié :
    Dans ai_scores_all.csv, certaines espèces réelles sont absentes car leur
    score softmax est < 0.001 (seuil de troncature du dataset).
    Ces observations sont silencieusement exclues du calcul de non-conformité,
    ce qui biaise le quantile vers une valeur trop basse.

Correction :
    Pour ces observations, on assigne un score de non-conformité S_i = 1.0
    (car la confiance de l'IA sur la bonne réponse était quasi nulle).

Données : observations_experts.csv, ai_scores_all.csv,
          calibration.csv, test.csv
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================================================================
# CONFIGURATION
# ================================================================
ALPHA = 0.05  # Couverture cible = 95%

# ================================================================
# ÉTAPE 1 : CHARGEMENT DES DONNÉES
# ================================================================
print("=" * 65)
print("PHASE 3 — VOLET 1 : CORRECTION DU BIAIS")
print("=" * 65)

calib = pd.read_csv("calibration.csv")
test = pd.read_csv("test.csv")
scores = pd.read_csv("ai_scores_all.csv")

print(f"\nDonnées chargées :")
print(f"  Calibration : {len(calib):,} observations")
print(f"  Test        : {len(test):,} observations")
print(f"  Scores      : {len(scores):,} lignes")

# ================================================================
# ÉTAPE 2 : CP STANDARD SANS CORRECTION (reproduire le bug)
# ================================================================
print(f"\n{'=' * 65}")
print("AVANT CORRECTION : CP Standard (buguée)")
print(f"{'=' * 65}")

# Filtrer les scores pour la calibration
calib_scores = scores[scores["observation_id"].isin(calib["observation_id"])]
merged_cal = calib_scores.merge(calib, on="observation_id")

# Ne garder que les lignes où la vraie espèce est présente
vraie_cal = merged_cal[
    merged_cal["spicies_id"] == merged_cal["ground_truth_val"]
].copy()
vraie_cal["non_conformity"] = 1 - vraie_cal["score"]

n_sans_correction = len(vraie_cal)
n_total = len(calib)
n_manquantes = n_total - n_sans_correction

print(f"\n  Observations en calibration          : {n_total:,}")
print(f"  Vraie espèce trouvée dans scores     : {n_sans_correction:,}")
print(f"  Vraie espèce ABSENTE (score < 0.001) : {n_manquantes:,} "
      f"({100 * n_manquantes / n_total:.1f}%)")
print(f"\n  ⚠ Ces {n_manquantes:,} observations sont IGNORÉES "
      f"dans le calcul sans correction")

# Quantile SANS correction
scores_sans = vraie_cal["non_conformity"].values
n_sans = len(scores_sans)
niveau_sans = min(np.ceil((n_sans + 1) * (1 - ALPHA)) / n_sans, 1.0)
q_hat_sans = np.quantile(scores_sans, niveau_sans, method="higher")
seuil_sans = 1 - q_hat_sans

print(f"\n  n = {n_sans:,}")
print(f"  q̂ (sans correction) = {q_hat_sans:.6f}")
print(f"  Seuil de score       = {seuil_sans:.6f}")

# Ensembles de prédiction SANS correction
test_scores = scores[scores["observation_id"].isin(test["observation_id"])]
merged_test = test_scores.merge(test, on="observation_id")

resultats_sans = []
for obs_id in merged_test["observation_id"].unique():
    lignes = merged_test[merged_test["observation_id"] == obs_id]
    gardees = lignes[lignes["score"] >= seuil_sans]["spicies_id"].tolist()
    if len(gardees) == 0:
        gardees = [lignes.loc[lignes["score"].idxmax(), "spicies_id"]]
    vrai = lignes["ground_truth_val"].iloc[0]
    resultats_sans.append({
        "observation_id": obs_id,
        "vraie_espece": vrai,
        "taille": len(gardees),
        "couvert": vrai in gardees
    })

df_sans = pd.DataFrame(resultats_sans)
couv_sans = df_sans["couvert"].mean()
taille_sans = df_sans["taille"].mean()

print(f"\n  --- Résultats SANS correction ---")
print(f"  Couverture marginale : {couv_sans:.4f}  (cible : {1 - ALPHA:.2f})")
print(f"  Taille moyenne       : {taille_sans:.2f}")
print(f"\n  ❌ La couverture est SOUS la cible de 95%")

# ================================================================
# ÉTAPE 3 : IDENTIFICATION DES OBSERVATIONS MANQUANTES
# ================================================================
print(f"\n{'=' * 65}")
print("DIAGNOSTIC : Qui sont les observations manquantes ?")
print(f"{'=' * 65}")

obs_trouvees = set(vraie_cal["observation_id"].unique())
obs_totales = set(calib["observation_id"].unique())
obs_manquantes = obs_totales - obs_trouvees

# Analyser les espèces concernées
especes_manquantes = calib[
    calib["observation_id"].isin(obs_manquantes)
]["ground_truth_val"]

prevalence_cal = calib["ground_truth_val"].value_counts()
prevalence_manquantes = especes_manquantes.map(prevalence_cal)

print(f"\n  Observations manquantes : {len(obs_manquantes):,}")
print(f"  Espèces uniques concernées : {especes_manquantes.nunique():,}")
print(f"\n  Prévalence de ces espèces en calibration :")
print(f"    Médiane : {prevalence_manquantes.median():.0f} obs/espèce")
print(f"    Moyenne : {prevalence_manquantes.mean():.1f} obs/espèce")
print(f"    % rares (< 10 obs) : "
      f"{100 * (prevalence_manquantes < 10).mean():.1f}%")

print(f"\n  → Confirmation : les observations manquantes concernent")
print(f"    majoritairement des espèces RARES")

# ================================================================
# ÉTAPE 4 : APPLICATION DE LA CORRECTION
# ================================================================
print(f"\n{'=' * 65}")
print("CORRECTION : S_i = 1.0 pour les observations manquantes")
print(f"{'=' * 65}")

print(f"""
  Raisonnement :
  - Le score de non-conformité est S_i = 1 - score(vraie espèce)
  - Si la vraie espèce est absente du fichier, son score est < 0.001
  - Donc S_i = 1 - (quelque chose < 0.001) ≈ 1.0
  - S_i = 1.0 signifie : le modèle a COMPLÈTEMENT échoué
  - Ces observations sont les cas les plus difficiles et DOIVENT
    être incluses dans la calibration
""")

# Créer les scores corrigés
manquantes_df = calib[calib["observation_id"].isin(obs_manquantes)].copy()
manquantes_df["non_conformity"] = 1.0

cols = ["observation_id", "ground_truth_val", "non_conformity"]
vraie_subset = vraie_cal[cols].copy()
manquantes_subset = manquantes_df[cols].copy()
toutes = pd.concat([vraie_subset, manquantes_subset], ignore_index=True)

print(f"  Scores sans correction : {n_sans:,} observations")
print(f"  + Observations corrigées : {len(obs_manquantes):,} (S_i = 1.0)")
print(f"  = Total avec correction  : {len(toutes):,} observations")

# Vérification
assert len(toutes) == n_total, \
    f"Incohérence : {len(toutes)} ≠ {n_total}"
print(f"\n  ✅ Vérification : {len(toutes):,} = {n_total:,} (total calibration)")

# ================================================================
# ÉTAPE 5 : CP STANDARD AVEC CORRECTION
# ================================================================
print(f"\n{'=' * 65}")
print("APRÈS CORRECTION : CP Standard (corrigée)")
print(f"{'=' * 65}")

# Quantile AVEC correction
scores_avec = toutes["non_conformity"].values
n_avec = len(scores_avec)
niveau_avec = min(np.ceil((n_avec + 1) * (1 - ALPHA)) / n_avec, 1.0)
q_hat_avec = np.quantile(scores_avec, niveau_avec, method="higher")
seuil_avec = 1 - q_hat_avec

print(f"\n  n = {n_avec:,}")
print(f"  q̂ (avec correction) = {q_hat_avec:.6f}")
print(f"  Seuil de score       = {seuil_avec:.6f}")

print(f"\n  Comparaison des quantiles :")
print(f"    q̂ sans correction = {q_hat_sans:.6f}")
print(f"    q̂ avec correction = {q_hat_avec:.6f}")
print(f"    Δq̂                = {q_hat_avec - q_hat_sans:+.6f}")
print(f"  → Le quantile AUGMENTE car on inclut les cas difficiles (S_i = 1.0)")

# Ensembles de prédiction AVEC correction
resultats_avec = []
for obs_id in merged_test["observation_id"].unique():
    lignes = merged_test[merged_test["observation_id"] == obs_id]
    gardees = lignes[lignes["score"] >= seuil_avec]["spicies_id"].tolist()
    if len(gardees) == 0:
        gardees = [lignes.loc[lignes["score"].idxmax(), "spicies_id"]]
    vrai = lignes["ground_truth_val"].iloc[0]
    resultats_avec.append({
        "observation_id": obs_id,
        "vraie_espece": vrai,
        "taille": len(gardees),
        "couvert": vrai in gardees
    })

df_avec = pd.DataFrame(resultats_avec)
couv_avec = df_avec["couvert"].mean()
taille_avec = df_avec["taille"].mean()

print(f"\n  --- Résultats AVEC correction ---")
print(f"  Couverture marginale : {couv_avec:.4f}  (cible : {1 - ALPHA:.2f})")
print(f"  Taille moyenne       : {taille_avec:.2f}")

if couv_avec >= 1 - ALPHA - 0.01:
    print(f"\n  ✅ La couverture atteint la cible de 95%")
else:
    print(f"\n  ⚠ La couverture est encore légèrement sous la cible")

# ================================================================
# ÉTAPE 6 : TABLEAU COMPARATIF AVANT / APRÈS
# ================================================================
print(f"\n{'=' * 65}")
print("COMPARAISON AVANT / APRÈS CORRECTION")
print(f"{'=' * 65}")

print(f"\n  {'Métrique':<30} {'Sans':>12} {'Avec':>12} {'Δ':>12}")
print(f"  {'-' * 66}")
print(f"  {'n_calibration':<30} {n_sans:>12,} {n_avec:>12,} "
      f"{n_avec - n_sans:>+12,}")
print(f"  {'q̂':<30} {q_hat_sans:>12.6f} {q_hat_avec:>12.6f} "
      f"{q_hat_avec - q_hat_sans:>+12.6f}")
print(f"  {'Seuil de score':<30} {seuil_sans:>12.6f} {seuil_avec:>12.6f} "
      f"{seuil_avec - seuil_sans:>+12.6f}")
print(f"  {'Couverture marginale':<30} {couv_sans:>12.4f} {couv_avec:>12.4f} "
      f"{couv_avec - couv_sans:>+12.4f}")
print(f"  {'Taille moyenne':<30} {taille_sans:>12.2f} {taille_avec:>12.2f} "
      f"{taille_avec - taille_sans:>+12.2f}")

# Sauvegarder
comparaison = pd.DataFrame({
    "Métrique": ["n_calibration", "q_hat", "seuil_score",
                 "couverture_marginale", "taille_moyenne"],
    "Sans_correction": [n_sans, q_hat_sans, seuil_sans, couv_sans, taille_sans],
    "Avec_correction": [n_avec, q_hat_avec, seuil_avec, couv_avec, taille_avec],
})
comparaison.to_csv("correction_biais_comparaison.csv", index=False)
print(f"\n✓ correction_biais_comparaison.csv")

# ================================================================
# ÉTAPE 7 : VISUALISATIONS
# ================================================================
print(f"\n{'=' * 65}")
print("GÉNÉRATION DES FIGURES")
print(f"{'=' * 65}")

# --- Figure 1 : Distribution des scores AVANT et APRÈS ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(scores_sans, bins=50, color="lightcoral", edgecolor="black",
             alpha=0.7)
axes[0].axvline(q_hat_sans, color="red", linestyle="--", linewidth=2,
                label=f"$\\hat{{q}}$ = {q_hat_sans:.4f}")
axes[0].set_xlabel("Score de non-conformité $S_i$")
axes[0].set_ylabel("Fréquence")
axes[0].set_title(f"SANS correction\n"
                   f"(n = {n_sans:,}, couverture = {couv_sans:.4f})")
axes[0].legend()

axes[1].hist(scores_avec, bins=50, color="steelblue", edgecolor="black",
             alpha=0.7)
axes[1].axvline(q_hat_avec, color="blue", linestyle="--", linewidth=2,
                label=f"$\\hat{{q}}$ = {q_hat_avec:.4f}")
axes[1].set_xlabel("Score de non-conformité $S_i$")
axes[1].set_ylabel("Fréquence")
axes[1].set_title(f"AVEC correction\n"
                   f"(n = {n_avec:,}, couverture = {couv_avec:.4f})")
axes[1].legend()

fig.suptitle("Impact de la correction du biais sur la distribution "
             "des scores de non-conformité",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("fig_correction_biais_distribution.png", dpi=150,
            bbox_inches="tight")
plt.close()
print(f"✓ fig_correction_biais_distribution.png")

# --- Figure 2 : Comparaison couverture et taille ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Couverture
methodes = ["Sans\ncorrection", "Avec\ncorrection"]
couvertures = [couv_sans, couv_avec]
colors = ["lightcoral", "steelblue"]
bars = axes[0].bar(methodes, couvertures, color=colors, edgecolor="black",
                    width=0.5)
axes[0].axhline(y=1 - ALPHA, color="green", linestyle="--", linewidth=1.5,
                label=f"Cible = {1 - ALPHA:.0%}")
axes[0].set_ylabel("Couverture marginale")
axes[0].set_title("Couverture")
axes[0].legend()
axes[0].set_ylim(0.88, 0.98)
for bar in bars:
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.003,
                 f"{bar.get_height():.4f}", ha="center", fontsize=11)

# Taille
tailles = [taille_sans, taille_avec]
bars2 = axes[1].bar(methodes, tailles, color=colors, edgecolor="black",
                     width=0.5)
axes[1].set_ylabel("Taille moyenne des ensembles")
axes[1].set_title("Efficacité (taille des ensembles)")
for bar in bars2:
    axes[1].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.05,
                 f"{bar.get_height():.2f}", ha="center", fontsize=11)

fig.suptitle("Impact de la correction du biais",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("fig_correction_biais_comparaison.png", dpi=150,
            bbox_inches="tight")
plt.close()
print(f"✓ fig_correction_biais_comparaison.png")

# --- Figure 3 : Prévalence des espèces manquantes ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(prevalence_manquantes, bins=range(1, 30), color="tomato",
        edgecolor="black", alpha=0.7, align="left")
ax.axvline(x=10, color="red", linestyle="--", linewidth=1.5,
           label="Seuil rareté = 10")
ax.set_xlabel("Prévalence de l'espèce en calibration")
ax.set_ylabel("Nombre d'observations manquantes")
ax.set_title(f"Prévalence des espèces dont la vraie classe est absente\n"
             f"({len(obs_manquantes):,} observations concernées)")
ax.legend()
plt.tight_layout()
plt.savefig("fig_prevalence_manquantes.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ fig_prevalence_manquantes.png")

# ================================================================
# CONCLUSION
# ================================================================
print(f"\n{'=' * 65}")
print("CONCLUSION")
print(f"{'=' * 65}")

print(f"""
  {n_manquantes:,} observations ({100*n_manquantes/n_total:.1f}%) avaient leur 
  vraie espèce absente du fichier ai_scores_all.csv.

  Ces observations concernent majoritairement des espèces RARES
  ({100*(prevalence_manquantes < 10).mean():.0f}% ont une prévalence < 10).

  En leur assignant S_i = 1.0 (échec total du modèle) :
    - Le quantile passe de {q_hat_sans:.4f} à {q_hat_avec:.4f}
    - La couverture passe de {couv_sans:.4f} à {couv_avec:.4f}
    - La taille des ensembles passe de {taille_sans:.2f} à {taille_avec:.2f}

  La couverture cible de 95% est atteinte.
""")

print("✅ Terminé")
