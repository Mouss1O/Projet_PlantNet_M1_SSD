"""
=============================================================================
Phase 3 : Méthode PAS (Partitioned Adaptive Sets)
=============================================================================
Auteur  : [Ton prénom] - Phase 3 du projet Pl@ntNet-CP
Objectif: Résoudre le problème de la Longue Traîne en calibrant
          séparément les espèces rares et communes.

Entrées :
    - fichiers_utilises/calibration.csv    (observation_id, ground_truth_val)
    - fichiers_utilises/test.csv           (observation_id, ground_truth_val)
    - fichiers_utilises/ai_scores_all.csv  (observation_id, spicies_id, score)
    - seuil.csv                            (résultat de la Phase 2 - baseline)

Sorties :
    - resultats_pas.csv        (ensembles de prédiction PAS)
    - comparaison_baseline_pas.csv (tableau comparatif)

Dépendances : pandas, numpy, matplotlib
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ================================================================
# CONFIGURATION
# ================================================================
ALPHA = 0.05                # Risque d'erreur (couverture cible = 95%)
SEUIL_RARETE = 10           # Espèce "Rare" si < 10 observations en calibration
RANDOM_STATE = 42

# Chemins — adaptés à la structure du dossier Méthode_PAS
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))       # Script_Used/
METHODE_PAS  = os.path.dirname(SCRIPT_DIR)                      # Méthode_PAS/
DIR_DONNEES  = os.path.join(METHODE_PAS, "Données")             # Données/
DIR_BASELINE = os.path.join(METHODE_PAS, "Resultats_Marginale") # Resultats baseline
DIR_RESULTATS = os.path.join(METHODE_PAS, "Resultats_PAS")      # Nos résultats

PATH_CAL    = os.path.join(DIR_DONNEES, "calibration.csv")
PATH_TEST   = os.path.join(DIR_DONNEES, "test.csv")
PATH_SCORES = os.path.join(DIR_DONNEES, "ai_scores_all.csv")
PATH_SEUIL  = os.path.join(DIR_DONNEES, "seuil.csv")


# ================================================================
# ÉTAPE 1 : CHARGEMENT DES DONNÉES
# ================================================================
def charger_donnees():
    """
    Charge les fichiers produits par les Phases 1 et 2.
    Retourne les DataFrames de calibration, test et scores.
    """
    print("=" * 60)
    print("PHASE 3 : MÉTHODE PAS (Partitioned Adaptive Sets)")
    print("=" * 60)

    # Vérification de l'existence des fichiers
    for path in [PATH_CAL, PATH_TEST, PATH_SCORES]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Fichier manquant : {path}\n"
                f"Assurez-vous d'avoir exécuté les Phases 1 et 2."
            )

    calib = pd.read_csv(PATH_CAL)
    test = pd.read_csv(PATH_TEST)
    scores = pd.read_csv(PATH_SCORES)

    print(f"Calibration : {len(calib):,} observations")
    print(f"Test        : {len(test):,} observations")
    print(f"Scores      : {len(scores):,} lignes")

    return calib, test, scores


# ================================================================
# ÉTAPE 2 : ANALYSE DE LA PRÉVALENCE (compter les obs par espèce)
# ================================================================
def analyser_prevalence(calib):
    """
    Compte le nombre d'observations par espèce dans le set de calibration.
    C'est la base pour définir les groupes "Rare" vs "Commun".

    Paramètres
    ----------
    calib : DataFrame avec colonnes [observation_id, ground_truth_val]

    Retourne
    --------
    prevalence : Series indexée par spicies_id, valeur = nombre d'observations
    """
    prevalence = calib["ground_truth_val"].value_counts()

    n_total = len(prevalence)
    n_rares = (prevalence < SEUIL_RARETE).sum()
    n_communes = (prevalence >= SEUIL_RARETE).sum()

    print(f"\n{'=' * 60}")
    print(f"ANALYSE DE LA PRÉVALENCE (seuil de rareté = {SEUIL_RARETE})")
    print(f"{'=' * 60}")
    print(f"Espèces totales      : {n_total:,}")
    print(f"Espèces RARES  (< {SEUIL_RARETE:>2}) : {n_rares:,}  "
          f"({100 * n_rares / n_total:.1f}%)")
    print(f"Espèces COMMUNES (>= {SEUIL_RARETE:>2}) : {n_communes:,}  "
          f"({100 * n_communes / n_total:.1f}%)")
    print(f"\nStatistiques de prévalence :")
    print(f"  Médiane : {prevalence.median():.0f} obs/espèce")
    print(f"  Moyenne : {prevalence.mean():.1f} obs/espèce")
    print(f"  Max     : {prevalence.max():,} obs/espèce")

    return prevalence


# ================================================================
# ÉTAPE 3 : ASSIGNATION DES GROUPES
# ================================================================
def assigner_groupes(prevalence):
    """
    Crée un dictionnaire {spicies_id -> "Rare" ou "Commun"}
    basé sur la prévalence dans le set de calibration.

    Paramètres
    ----------
    prevalence : Series (index=spicies_id, valeur=count)

    Retourne
    --------
    groupes : dict {spicies_id: str}
    """
    groupes = {}
    for espece, count in prevalence.items():
        if count < SEUIL_RARETE:
            groupes[espece] = "Rare"
        else:
            groupes[espece] = "Commun"

    return groupes


# ================================================================
# ÉTAPE 4 : CALCUL DES SCORES DE NON-CONFORMITÉ PAR GROUPE
# ================================================================
def calculer_non_conformite_par_groupe(calib, scores, groupes):
    """
    Pour chaque observation de calibration :
    1. Trouve le score softmax attribué à la vraie espèce
    2. Calcule s_i = 1 - score(vraie espèce)
    3. Si la vraie espèce est ABSENTE des scores (score < 0.001),
       on assigne s_i = 1.0 (échec total du modèle)
    4. Assigne le groupe (Rare/Commun) selon la vraie espèce

    CORRECTION IMPORTANTE : Ne pas ignorer les observations où le modèle
    n'a pas scoré la vraie espèce. Ces cas sont les plus difficiles et
    doivent être inclus pour que le quantile garantisse la couverture.

    Retourne
    --------
    scores_rare : array des scores de non-conformité des espèces rares
    scores_commun : array des scores de non-conformité des espèces communes
    """
    # Filtrer les scores pour les observations de calibration
    calib_scores = scores[scores["observation_id"].isin(calib["observation_id"])]

    # Fusionner pour avoir le ground_truth à côté
    merged = calib_scores.merge(calib, on="observation_id")

    # Garder uniquement la ligne où l'espèce prédite = vraie espèce
    vraie = merged[merged["spicies_id"] == merged["ground_truth_val"]].copy()
    vraie["non_conformity"] = 1 - vraie["score"]

    # --- CORRECTION : observations où la vraie espèce est absente ---
    obs_trouvees = set(vraie["observation_id"].unique())
    obs_totales = set(calib["observation_id"].unique())
    obs_manquantes = obs_totales - obs_trouvees

    n_manquantes = len(obs_manquantes)
    n_totales = len(obs_totales)

    print(f"\n{'=' * 60}")
    print("SCORES DE NON-CONFORMITÉ PAR GROUPE")
    print(f"{'=' * 60}")
    print(f"Observations totales en calibration : {n_totales:,}")
    print(f"  Vraie espèce trouvée dans scores  : {len(obs_trouvees):,}")
    print(f"  Vraie espèce ABSENTE (score < 0.001) : {n_manquantes:,} "
          f"({100 * n_manquantes / n_totales:.1f}%)")
    print(f"  → Ces {n_manquantes:,} observations reçoivent s_i = 1.0")

    # Créer les lignes manquantes avec non_conformity = 1.0
    if n_manquantes > 0:
        manquantes_df = calib[calib["observation_id"].isin(obs_manquantes)].copy()
        manquantes_df["non_conformity"] = 1.0
        manquantes_df["spicies_id"] = manquantes_df["ground_truth_val"]
        manquantes_df["score"] = 0.0

        # Combiner avec les observations trouvées
        cols_communs = ["observation_id", "ground_truth_val", "non_conformity"]
        vraie_subset = vraie[cols_communs].copy()
        manquantes_subset = manquantes_df[cols_communs].copy()
        toutes = pd.concat([vraie_subset, manquantes_subset], ignore_index=True)
    else:
        toutes = vraie[["observation_id", "ground_truth_val",
                        "non_conformity"]].copy()

    # Assigner le groupe
    toutes["groupe"] = toutes["ground_truth_val"].map(groupes)

    # Séparer par groupe
    scores_rare = toutes[toutes["groupe"] == "Rare"]["non_conformity"].values
    scores_commun = toutes[toutes["groupe"] == "Commun"]["non_conformity"].values

    print(f"\nObservations Rares    : {len(scores_rare):,}")
    print(f"  Moyenne s_i : {scores_rare.mean():.4f}")
    print(f"  Médiane s_i : {np.median(scores_rare):.4f}")
    print(f"Observations Communes : {len(scores_commun):,}")
    print(f"  Moyenne s_i : {scores_commun.mean():.4f}")
    print(f"  Médiane s_i : {np.median(scores_commun):.4f}")

    # Vérification : les rares ont des scores plus élevés (modèle moins sûr)
    if scores_rare.mean() > scores_commun.mean():
        print("\n→ Confirmé : le modèle est MOINS confiant sur les espèces rares")
    else:
        print("\n⚠ Inattendu : le modèle semble aussi confiant sur les rares")

    return scores_rare, scores_commun, toutes


# ================================================================
# ÉTAPE 5 : CALCUL DES QUANTILES ADAPTATIFS (un par groupe)
# ================================================================
def calculer_quantiles_adaptatifs(scores_rare, scores_commun):
    """
    Calcule un quantile séparé pour chaque groupe.
    Même formule que la Phase 2, mais appliquée indépendamment.

    q̂_groupe = quantile( scores_groupe, niveau = (1-α)(1 + 1/n_groupe) )

    Retourne
    --------
    q_hat_rare : float   - quantile pour les espèces rares
    q_hat_commun : float - quantile pour les espèces communes
    """
    n_rare = len(scores_rare)
    n_commun = len(scores_commun)

    # Niveaux de quantile corrigés (correction de continuité)
    niveau_rare = (1 - ALPHA) * (1 + 1 / n_rare)
    niveau_commun = (1 - ALPHA) * (1 + 1 / n_commun)

    # Sécurité : le niveau ne peut pas dépasser 1
    niveau_rare = min(niveau_rare, 1.0)
    niveau_commun = min(niveau_commun, 1.0)

    q_hat_rare = np.quantile(scores_rare, niveau_rare, method="higher")
    q_hat_commun = np.quantile(scores_commun, niveau_commun, method="higher")

    seuil_rare = 1 - q_hat_rare
    seuil_commun = 1 - q_hat_commun

    print(f"\n{'=' * 60}")
    print("QUANTILES ADAPTATIFS (PAS)")
    print(f"{'=' * 60}")
    print(f"Groupe RARE :")
    print(f"  n = {n_rare:,}")
    print(f"  Niveau du quantile = {niveau_rare:.6f}")
    print(f"  q̂_rare   = {q_hat_rare:.6f}")
    print(f"  → Seuil  = {seuil_rare:.6f}  (on garde score >= {seuil_rare:.6f})")
    print(f"\nGroupe COMMUN :")
    print(f"  n = {n_commun:,}")
    print(f"  Niveau du quantile = {niveau_commun:.6f}")
    print(f"  q̂_commun = {q_hat_commun:.6f}")
    print(f"  → Seuil  = {seuil_commun:.6f}  (on garde score >= {seuil_commun:.6f})")

    if q_hat_rare > q_hat_commun:
        print(f"\n→ Confirmé : le seuil des rares est PLUS PRUDENT "
              f"(accepte plus de candidats)")

    # Charger le seuil baseline pour comparaison
    if os.path.exists(PATH_SEUIL):
        baseline = pd.read_csv(PATH_SEUIL)
        q_hat_baseline = baseline["q_hat"].iloc[0]
        print(f"\n--- Comparaison avec la Baseline (Phase 2) ---")
        print(f"  q̂_baseline = {q_hat_baseline:.6f}")
        print(f"  q̂_rare     = {q_hat_rare:.6f}  "
              f"({'↑' if q_hat_rare > q_hat_baseline else '↓'} vs baseline)")
        print(f"  q̂_commun   = {q_hat_commun:.6f}  "
              f"({'↑' if q_hat_commun > q_hat_baseline else '↓'} vs baseline)")

    return q_hat_rare, q_hat_commun


# ================================================================
# ÉTAPE 6 : CONSTRUCTION DES ENSEMBLES DE PRÉDICTION PAS
# ================================================================
def construire_ensembles_pas(test, scores, groupes, q_hat_rare, q_hat_commun):
    """
    Pour chaque image du test set, construit l'ensemble de prédiction PAS.

    Pour chaque espèce candidate y :
        - Si y est "Rare"   → on la garde si score >= 1 - q̂_rare
        - Si y est "Commun" → on la garde si score >= 1 - q̂_commun
        - Si y est inconnue (pas dans le calibration set) → on utilise q̂_rare
          (principe de précaution)

    Retourne
    --------
    resultats : DataFrame avec colonnes
        [observation_id, vraie_espece, groupe_vrai, ensemble, taille, couvert]
    """
    seuil_rare = 1 - q_hat_rare
    seuil_commun = 1 - q_hat_commun

    print(f"\n{'=' * 60}")
    print("CONSTRUCTION DES ENSEMBLES DE PRÉDICTION PAS")
    print(f"{'=' * 60}")
    print(f"Seuil Rare   : score >= {seuil_rare:.6f}")
    print(f"Seuil Commun : score >= {seuil_commun:.6f}")

    # Filtrer les scores pour les observations de test
    test_scores = scores[scores["observation_id"].isin(test["observation_id"])]
    merged = test_scores.merge(test, on="observation_id")

    resultats = []
    obs_ids = merged["observation_id"].unique()

    for i, obs_id in enumerate(obs_ids):
        if (i + 1) % 500 == 0:
            print(f"  Traitement : {i + 1:,} / {len(obs_ids):,}")

        lignes = merged[merged["observation_id"] == obs_id]

        # Pour chaque espèce candidate, appliquer le seuil de son groupe
        gardees = []
        for _, row in lignes.iterrows():
            espece = row["spicies_id"]
            score_val = row["score"]

            # Déterminer le groupe de cette espèce candidate
            groupe_espece = groupes.get(espece, "Rare")  # Prudence par défaut

            # Appliquer le seuil du groupe
            if groupe_espece == "Rare" and score_val >= seuil_rare:
                gardees.append(espece)
            elif groupe_espece == "Commun" and score_val >= seuil_commun:
                gardees.append(espece)

        # Si l'ensemble est vide, garder au moins la meilleure prédiction
        if len(gardees) == 0:
            meilleure = lignes.loc[lignes["score"].idxmax(), "spicies_id"]
            gardees = [meilleure]

        vrai = lignes["ground_truth_val"].iloc[0]
        groupe_vrai = groupes.get(vrai, "Rare")

        resultats.append({
            "observation_id": obs_id,
            "vraie_espece": vrai,
            "groupe_vrai": groupe_vrai,
            "ensemble": gardees,
            "taille": len(gardees),
            "couvert": vrai in gardees,
        })

    res_df = pd.DataFrame(resultats)
    print(f"\n  Observations traitées : {len(res_df):,}")

    return res_df


# ================================================================
# ÉTAPE 7 : ÉVALUATION ET COMPARAISON
# ================================================================
def evaluer_et_comparer(res_pas):
    """
    Calcule les métriques de performance de PAS et compare avec la Baseline.
    """
    print(f"\n{'=' * 60}")
    print("RÉSULTATS PAS")
    print(f"{'=' * 60}")

    # --- Métriques globales PAS ---
    couv_globale = res_pas["couvert"].mean()
    taille_moy = res_pas["taille"].mean()
    print(f"Couverture globale PAS  : {couv_globale:.4f}  "
          f"(cible : {1 - ALPHA:.2f})")
    print(f"Taille moyenne PAS      : {taille_moy:.2f}")

    # --- Métriques par groupe PAS ---
    print(f"\n--- Couverture par groupe (PAS) ---")
    for groupe in ["Rare", "Commun"]:
        sub = res_pas[res_pas["groupe_vrai"] == groupe]
        if len(sub) > 0:
            couv = sub["couvert"].mean()
            taille = sub["taille"].mean()
            print(f"  {groupe:>7s} : couverture = {couv:.4f}, "
                  f"taille moy = {taille:.2f}, "
                  f"n = {len(sub):,}")

    # --- Comparaison avec Baseline ---
    path_baseline = os.path.join(DIR_BASELINE, "resultats.csv")
    if os.path.exists(path_baseline):
        print(f"\n--- COMPARAISON BASELINE vs PAS ---")
        res_baseline = pd.read_csv(path_baseline)

        # Recalculer le groupe pour la baseline
        # (la baseline n'a pas de colonne groupe_vrai)
        groupes_map = dict(
            zip(res_pas["vraie_espece"], res_pas["groupe_vrai"])
        )

        # S'assurer que les mêmes observations sont comparées
        common_ids = set(res_baseline["observation_id"]) & set(
            res_pas["observation_id"]
        )
        bl = res_baseline[res_baseline["observation_id"].isin(common_ids)].copy()
        bl["groupe_vrai"] = bl["vraie_espece"].map(groupes_map)

        print(f"\n  {'Métrique':<25} {'Baseline':>10} {'PAS':>10} {'Δ':>10}")
        print(f"  {'-' * 55}")

        # Global
        couv_bl = bl["couvert"].mean()
        couv_pas = res_pas["couvert"].mean()
        print(f"  {'Couverture globale':<25} {couv_bl:>10.4f} {couv_pas:>10.4f} "
              f"{couv_pas - couv_bl:>+10.4f}")

        taille_bl = bl["taille"].mean()
        taille_pas = res_pas["taille"].mean()
        print(f"  {'Taille moyenne':<25} {taille_bl:>10.2f} {taille_pas:>10.2f} "
              f"{taille_pas - taille_bl:>+10.2f}")

        # Par groupe
        for groupe in ["Rare", "Commun"]:
            sub_bl = bl[bl["groupe_vrai"] == groupe]
            sub_pas = res_pas[res_pas["groupe_vrai"] == groupe]
            if len(sub_bl) > 0 and len(sub_pas) > 0:
                c_bl = sub_bl["couvert"].mean()
                c_pas = sub_pas["couvert"].mean()
                t_bl = sub_bl["taille"].mean()
                t_pas = sub_pas["taille"].mean()
                print(f"  {'Couv. ' + groupe:<25} {c_bl:>10.4f} {c_pas:>10.4f} "
                      f"{c_pas - c_bl:>+10.4f}")
                print(f"  {'Taille ' + groupe:<25} {t_bl:>10.2f} {t_pas:>10.2f} "
                      f"{t_pas - t_bl:>+10.2f}")

        # Sauvegarder la comparaison
        comparaison = pd.DataFrame({
            "Methode": ["Baseline", "PAS"],
            "Couverture_globale": [couv_bl, couv_pas],
            "Taille_moyenne": [taille_bl, taille_pas],
        })
        comparaison.to_csv(os.path.join(DIR_RESULTATS, "comparaison_baseline_pas.csv"), index=False)
        print(f"\n  Fichier sauvegardé : comparaison_baseline_pas.csv")
    else:
        print("\n⚠ Fichier resultats.csv (Baseline) non trouvé - "
              "comparaison impossible")

    return couv_globale, taille_moy


# ================================================================
# ÉTAPE 8 : VISUALISATIONS
# ================================================================
def generer_graphiques(res_pas, scores_rare, scores_commun,
                       q_hat_rare, q_hat_commun):
    """
    Génère les graphiques pour le rapport.
    """
    print(f"\n{'=' * 60}")
    print("GÉNÉRATION DES GRAPHIQUES")
    print(f"{'=' * 60}")

    DIR_FIGURES = os.path.join(DIR_RESULTATS, "figures")
    os.makedirs(DIR_FIGURES, exist_ok=True)

    # --- Figure 1 : Distribution des scores de non-conformité par groupe ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(scores_rare, bins=50, alpha=0.7, color="tomato",
                 edgecolor="black", label="Rares")
    axes[0].axvline(q_hat_rare, color="red", linestyle="--", linewidth=2,
                    label=f"$\\hat{{q}}_{{rare}}$ = {q_hat_rare:.4f}")
    axes[0].set_xlabel("Score de non-conformité $s_i$")
    axes[0].set_ylabel("Fréquence")
    axes[0].set_title("Espèces Rares")
    axes[0].legend()

    axes[1].hist(scores_commun, bins=50, alpha=0.7, color="steelblue",
                 edgecolor="black", label="Communes")
    axes[1].axvline(q_hat_commun, color="blue", linestyle="--", linewidth=2,
                    label=f"$\\hat{{q}}_{{commun}}$ = {q_hat_commun:.4f}")
    axes[1].set_xlabel("Score de non-conformité $s_i$")
    axes[1].set_ylabel("Fréquence")
    axes[1].set_title("Espèces Communes")
    axes[1].legend()

    fig.suptitle("Distribution des scores de non-conformité par groupe",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_FIGURES, "fig1_distribution_non_conformite.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  ✓ fig1_distribution_non_conformite.png")

    # --- Figure 2 : Comparaison couverture Baseline vs PAS ---
    path_baseline = os.path.join(DIR_BASELINE, "resultats.csv")
    if os.path.exists(path_baseline):
        res_baseline = pd.read_csv(path_baseline)
        groupes_map = dict(
            zip(res_pas["vraie_espece"], res_pas["groupe_vrai"])
        )
        res_baseline["groupe_vrai"] = res_baseline["vraie_espece"].map(
            groupes_map
        )

        categories = ["Global", "Rares", "Communes"]
        couv_bl = [
            res_baseline["couvert"].mean(),
            res_baseline[res_baseline["groupe_vrai"] == "Rare"]["couvert"].mean(),
            res_baseline[res_baseline["groupe_vrai"] == "Commun"]["couvert"].mean(),
        ]
        couv_pas = [
            res_pas["couvert"].mean(),
            res_pas[res_pas["groupe_vrai"] == "Rare"]["couvert"].mean(),
            res_pas[res_pas["groupe_vrai"] == "Commun"]["couvert"].mean(),
        ]

        x = np.arange(len(categories))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))
        bars1 = ax.bar(x - width / 2, couv_bl, width, label="Baseline",
                       color="lightcoral", edgecolor="black")
        bars2 = ax.bar(x + width / 2, couv_pas, width, label="PAS",
                       color="steelblue", edgecolor="black")

        ax.axhline(y=1 - ALPHA, color="green", linestyle="--", linewidth=1.5,
                   label=f"Cible = {1 - ALPHA:.0%}")
        ax.set_ylabel("Couverture")
        ax.set_title("Couverture : Baseline vs PAS")
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.set_ylim(0.85, 1.0)

        # Ajouter les valeurs sur les barres
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{bar.get_height():.3f}", ha="center", va="bottom",
                    fontsize=9)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{bar.get_height():.3f}", ha="center", va="bottom",
                    fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(DIR_FIGURES, "fig2_couverture_baseline_vs_pas.png"), dpi=150,
                    bbox_inches="tight")
        plt.close()
        print("  ✓ fig2_couverture_baseline_vs_pas.png")

    # --- Figure 3 : Distribution de la taille des ensembles par groupe ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, groupe in enumerate(["Rare", "Commun"]):
        sub = res_pas[res_pas["groupe_vrai"] == groupe]
        tailles = sub["taille"].value_counts().sort_index()
        axes[i].bar(tailles.index, tailles.values,
                    color="tomato" if groupe == "Rare" else "steelblue",
                    edgecolor="black", alpha=0.7)
        axes[i].set_xlabel("Taille de l'ensemble C(x)")
        axes[i].set_ylabel("Nombre d'observations")
        axes[i].set_title(f"Espèces {groupe}s\n"
                          f"(taille moy = {sub['taille'].mean():.2f})")

    fig.suptitle("Distribution de la taille des ensembles de prédiction (PAS)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_FIGURES, "fig3_taille_ensembles_pas.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  ✓ fig3_taille_ensembles_pas.png")


# ================================================================
# PROGRAMME PRINCIPAL
# ================================================================
def main():
    # Créer le dossier de résultats s'il n'existe pas
    os.makedirs(DIR_RESULTATS, exist_ok=True)

    # Étape 1 : Charger les données
    calib, test, scores = charger_donnees()

    # Étape 2 : Analyser la prévalence
    prevalence = analyser_prevalence(calib)

    # Étape 3 : Assigner les groupes
    groupes = assigner_groupes(prevalence)

    # Étape 4 : Calculer les scores de non-conformité par groupe
    scores_rare, scores_commun, df_nc = calculer_non_conformite_par_groupe(
        calib, scores, groupes
    )

    # Étape 5 : Calculer les quantiles adaptatifs
    q_hat_rare, q_hat_commun = calculer_quantiles_adaptatifs(
        scores_rare, scores_commun
    )

    # Étape 6 : Construire les ensembles de prédiction PAS
    res_pas = construire_ensembles_pas(
        test, scores, groupes, q_hat_rare, q_hat_commun
    )

    # Sauvegarder les résultats
    res_pas.to_csv(os.path.join(DIR_RESULTATS, "resultats_pas.csv"), index=False)
    print(f"\n  Fichier sauvegardé : resultats_pas.csv")

    # Étape 7 : Évaluer et comparer
    evaluer_et_comparer(res_pas)

    # Étape 8 : Graphiques
    generer_graphiques(res_pas, scores_rare, scores_commun,
                       q_hat_rare, q_hat_commun)

    print(f"\n{'=' * 60}")
    print("PHASE 3 TERMINÉE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrompu")
    except Exception as e:
        print(f"\nERREUR : {e}")
        import traceback
        traceback.print_exc()
