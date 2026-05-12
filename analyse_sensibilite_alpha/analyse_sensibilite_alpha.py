"""
=============================================================================
Partie 3 - Phase 1 : Analyse de sensibilite sur alpha (multi-seuils)
=============================================================================
Auteur  : Moussa Diagne
Objectif: Tester CP Standard et PAS sur plusieurs
          valeurs de alpha, avec correction du biais.

PAS : score rescale par la prevalence
    s_i = 1 - p(y_i | x_i) / p_hat(y_i)
    
    Un seul quantile global, applique sur les scores rescales.
    Les especes rares sont remontees par le facteur 1 / p_hat(y).

Combine :
    - CP Standard (Firda)
    - Correction du biais (Dossou) : s_i = 1.0 si vraie espece absente
    - PAS = score rescale par la prevalence (blog du prof)

Donnees : calibration.csv, test.csv, ai_scores_all.csv
=============================================================================
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ================================================================
# CHEMINS RELATIFS
# ================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.dirname(SCRIPT_DIR)

PATH_CAL    = os.path.join(ROOT, "data", "split_random", "calibration.csv")
PATH_TEST   = os.path.join(ROOT, "data", "split_random", "test.csv")
PATH_SCORES = os.path.join(ROOT, "Données", "ai_scores", "ai_scores_all.csv")

DIR_FIG = os.path.join(SCRIPT_DIR, "figures")
DIR_OUT = SCRIPT_DIR

os.makedirs(DIR_FIG, exist_ok=True)

# ================================================================
# CONFIGURATION
# ================================================================
ALPHA_GRID = [0.01, 0.05, 0.10, 0.20]


# ================================================================
# ETAPE 1 : CHARGEMENT DES DONNEES (avec gestion memoire)
# ================================================================
def charger_donnees():
    print("=" * 65)
    print("PARTIE 3 - PHASE 1 : ANALYSE DE SENSIBILITE SUR ALPHA")
    print("                     Avec vrai PAS (score rescale)")
    print("=" * 65)

    for p in [PATH_CAL, PATH_TEST, PATH_SCORES]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Fichier manquant : {p}")

    print("\nChargement calibration et test...")
    calib = pd.read_csv(PATH_CAL)
    test = pd.read_csv(PATH_TEST)
    print(f"  Calibration : {len(calib):,} observations")
    print(f"  Test        : {len(test):,} observations")

    print(f"\nChargement scores par chunks (fichier 2.1 GB)...")
    obs_interessantes = set(calib["observation_id"]) | set(test["observation_id"])
    print(f"  Observations a recuperer : {len(obs_interessantes):,}")

    chunksize = 500_000
    matched = []
    n_chunks = 0

    for chunk in pd.read_csv(PATH_SCORES, chunksize=chunksize,
                              dtype={"observation_id": np.int64,
                                     "spicies_id": np.int32,
                                     "score": np.float32}):
        mask = chunk["observation_id"].isin(obs_interessantes)
        if mask.any():
            matched.append(chunk[mask])
        n_chunks += 1
        if n_chunks % 20 == 0:
            print(f"  ... {n_chunks * chunksize:>10,} lignes lues")

    scores = pd.concat(matched, ignore_index=True)
    del matched
    print(f"\n  Scores filtres : {len(scores):,} lignes")

    return calib, test, scores


# ================================================================
# ETAPE 2 : CALCUL DE LA PREVALENCE 
# ================================================================
def calculer_prevalence(calib):
    """
    Calcule p_hat(y) = (1/N) * #{i : y_i = y}
    pour chaque espece y presente en calibration.
    """
    print(f"\n{'=' * 65}")
    print("CALCUL DE LA PREVALENCE p_hat(y)")
    print(f"{'=' * 65}")

    N = len(calib)
    comptes = calib["ground_truth_val"].value_counts()
    prevalence = comptes / N

    print(f"\n  N (taille calibration) = {N:,}")
    print(f"  Especes uniques        = {len(prevalence):,}")
    print(f"\n  Prevalence min  = {prevalence.min():.6f} "
          f"(1 obs sur {N:,})")
    print(f"  Prevalence max  = {prevalence.max():.6f} "
          f"({int(prevalence.max() * N):,} obs)")
    print(f"  Prevalence mediane = {prevalence.median():.6f}")

    return prevalence


# ================================================================
# ETAPE 3 : SCORES DE NON-CONFORMITE (avec correction biais)
# ================================================================
def calculer_scores_standard(calib, scores):
    """
    CP Standard : s_i = 1 - p(y_i | x_i)
    Avec correction : s_i = 1.0 si vraie espece absente
    """
    print(f"\n{'=' * 65}")
    print("SCORES CP STANDARD : s = 1 - p(y|x)")
    print(f"{'=' * 65}")

    # Filtrer les scores pour la calibration
    calib_scores = scores[scores["observation_id"].isin(calib["observation_id"])]
    merged_cal = calib_scores.merge(calib, on="observation_id")

    # Lignes ou l'espece predite = vraie espece
    vraie_cal = merged_cal[
        merged_cal["spicies_id"] == merged_cal["ground_truth_val"]
    ].copy()
    vraie_cal["s_std"] = 1.0 - vraie_cal["score"]

    # Identifier les observations manquantes
    obs_trouvees = set(vraie_cal["observation_id"].unique())
    obs_totales = set(calib["observation_id"].unique())
    obs_manquantes = obs_totales - obs_trouvees
    n_manquantes = len(obs_manquantes)

    print(f"\n  Obs totales       : {len(obs_totales):,}")
    print(f"  Obs trouvees      : {len(obs_trouvees):,}")
    print(f"  Obs manquantes    : {n_manquantes:,} -> s = 1.0 (correction biais)")

    # Creer les lignes manquantes
    manquantes_df = calib[calib["observation_id"].isin(obs_manquantes)].copy()
    manquantes_df["s_std"] = 1.0

    cols = ["observation_id", "ground_truth_val", "s_std"]
    df_std = pd.concat(
        [vraie_cal[cols], manquantes_df[cols]],
        ignore_index=True
    )

    assert len(df_std) == len(calib), "Incoherence Standard"
    print(f"  Total final       : {len(df_std):,}")

    return df_std


def calculer_scores_pas(df_std, prevalence):
    """
    Vrai PAS du blog du prof :
    
        s_i = 1 - p(y_i | x_i) / p_hat(y_i)
        
    On part du score Standard et on rescale par la prevalence.
    p(y|x) = 1 - s_std (si s_std vient d'une vraie observation)
    
    Pour les obs avec s_std = 1.0 (vraie espece absente),
    on garde s_pas = 1.0 (correction biais identique).
    """
    print(f"\n{'=' * 65}")
    print("SCORES PAS : s = 1 - p(y|x) / p_hat(y)")
    print(f"{'=' * 65}")

    df_pas = df_std.copy()

    # Recuperer la prevalence de la vraie espece de chaque obs
    df_pas["p_hat"] = df_pas["ground_truth_val"].map(prevalence)

    # Pour les obs avec correction biais (s_std = 1.0)
    # on a p(y|x) tres faible, donc p(y|x) / p_hat reste tres faible
    # donc s_pas reste tres proche de 1.0
    # On garde s_pas = 1.0 par convention
    is_correction = (df_pas["s_std"] == 1.0)

    # Pour les autres : p(y|x) = 1 - s_std
    p_y_given_x = 1.0 - df_pas["s_std"]
    score_rescale = p_y_given_x / df_pas["p_hat"]
    df_pas["s_pas"] = 1.0 - score_rescale

    # Pour les corrections, on impose s_pas = 1.0
    df_pas.loc[is_correction, "s_pas"] = 1.0

    # Note : s_pas peut etre NEGATIF si p(y|x) > p_hat(y)
    # Cela arrive quand le modele est tres confiant sur une espece rare
    # C'est normal et la theorie le gere

    print(f"\n  s_std : moyenne = {df_pas['s_std'].mean():.4f}, "
          f"min = {df_pas['s_std'].min():.4f}, "
          f"max = {df_pas['s_std'].max():.4f}")
    print(f"  s_pas : moyenne = {df_pas['s_pas'].mean():.4f}, "
          f"min = {df_pas['s_pas'].min():.4f}, "
          f"max = {df_pas['s_pas'].max():.4f}")
    print(f"\n  Observations avec s_pas < 0 : "
          f"{(df_pas['s_pas'] < 0).sum():,}")
    print(f"  (modele plus confiant que la prevalence -> score negatif)")

    return df_pas


# ================================================================
# ETAPE 4 : QUANTILE CONFORMEL
# ================================================================
def calculer_quantile(scores_array, alpha):
    """Quantile conformel avec correction finie-sample."""
    n = len(scores_array)
    niveau = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    q_hat = np.quantile(scores_array, niveau, method="higher")
    return float(q_hat)


# ================================================================
# ETAPE 5 : CONSTRUCTION DES ENSEMBLES - STANDARD
# ================================================================
def construire_ensembles_standard(test, scores, q_hat_std):
    """
    CP Standard : C(x) = { y : p(y|x) >= 1 - q_hat_std }
    """
    seuil = 1.0 - q_hat_std
    test_scores = scores[scores["observation_id"].isin(test["observation_id"])]
    merged = test_scores.merge(test, on="observation_id")

    resultats = []
    for obs_id in merged["observation_id"].unique():
        lignes = merged[merged["observation_id"] == obs_id]
        gardees = lignes[lignes["score"] >= seuil]["spicies_id"].tolist()
        if len(gardees) == 0:
            gardees = [lignes.loc[lignes["score"].idxmax(), "spicies_id"]]
        vrai = lignes["ground_truth_val"].iloc[0]
        resultats.append({
            "observation_id": obs_id,
            "vraie_espece": vrai,
            "taille": len(gardees),
            "couvert": vrai in gardees,
        })

    return pd.DataFrame(resultats)


# ================================================================
# ETAPE 6 : CONSTRUCTION DES ENSEMBLES - VRAI PAS
# ================================================================
def construire_ensembles_pas(test, scores, q_hat_pas, prevalence):
    """
    Vrai PAS :
        C(x) = { y : p(y|x) / p_hat(y) >= 1 - q_hat_pas }
    
    Pour chaque espece candidate, on calcule son score rescale
    en utilisant sa propre prevalence.
    """
    seuil_rescale = 1.0 - q_hat_pas
    test_scores = scores[scores["observation_id"].isin(test["observation_id"])]
    merged = test_scores.merge(test, on="observation_id")

    # Ajouter la prevalence de chaque espece candidate
    # Si l'espece n'est pas dans la calibration, on lui donne
    # une prevalence tres petite (= 1 obs / N) par prudence
    N_cal = len(test)  # taille de reference
    prev_min = 1.0 / N_cal
    merged["p_hat_candidat"] = merged["spicies_id"].map(prevalence).fillna(prev_min)

    # Score rescale pour chaque candidat
    merged["score_rescale"] = merged["score"] / merged["p_hat_candidat"]

    resultats = []
    for obs_id in merged["observation_id"].unique():
        lignes = merged[merged["observation_id"] == obs_id]
        gardees = lignes[
            lignes["score_rescale"] >= seuil_rescale
        ]["spicies_id"].tolist()

        if len(gardees) == 0:
            gardees = [lignes.loc[lignes["score_rescale"].idxmax(), "spicies_id"]]

        vrai = lignes["ground_truth_val"].iloc[0]
        resultats.append({
            "observation_id": obs_id,
            "vraie_espece": vrai,
            "taille": len(gardees),
            "couvert": vrai in gardees,
        })

    return pd.DataFrame(resultats)


# ================================================================
# ETAPE 7 : METRIQUES
# ================================================================
def calculer_metriques(df_resultats):
    """Couverture marginale, macro et taille moyenne."""
    couv_marg = df_resultats["couvert"].mean()
    taille_moy = df_resultats["taille"].mean()
    couv_par_espece = df_resultats.groupby("vraie_espece")["couvert"].mean()
    couv_macro = couv_par_espece.mean()
    return couv_marg, couv_macro, taille_moy


# ================================================================
# ETAPE 8 : BOUCLE PRINCIPALE
# ================================================================
def boucle_alpha(df_std, df_pas, test, scores, prevalence):
    print(f"\n{'=' * 65}")
    print("BOUCLE SUR LES VALEURS DE ALPHA")
    print(f"{'=' * 65}")

    scores_std = df_std["s_std"].values
    scores_pas = df_pas["s_pas"].values

    resultats = []
    for alpha in ALPHA_GRID:
        cible = 1 - alpha
        print(f"\n  {'-' * 60}")
        print(f"  ALPHA = {alpha}  (cible = {cible:.0%})")
        print(f"  {'-' * 60}")

        # --- Standard CP ---
        q_std = calculer_quantile(scores_std, alpha)
        res_std = construire_ensembles_standard(test, scores, q_std)
        marg_std, macro_std, taille_std = calculer_metriques(res_std)

        print(f"    Standard CP :")
        print(f"      q_hat            = {q_std:.4f}")
        print(f"      Couv. marginale  = {marg_std:.4f}")
        print(f"      Couv. macro      = {macro_std:.4f}")
        print(f"      Taille moyenne   = {taille_std:.2f}")

        # --- PAS ---
        q_pas = calculer_quantile(scores_pas, alpha)
        res_pas = construire_ensembles_pas(test, scores, q_pas, prevalence)
        marg_pas, macro_pas, taille_pas = calculer_metriques(res_pas)

        print(f"    PAS :")
        print(f"      q_hat            = {q_pas:.4f}")
        print(f"      Couv. marginale  = {marg_pas:.4f}")
        print(f"      Couv. macro      = {macro_pas:.4f}")
        print(f"      Taille moyenne   = {taille_pas:.2f}")

        # Compare macro
        gain_macro = macro_pas - macro_std
        symbole = "OK" if gain_macro >= 0 else "X"
        print(f"\n    [{symbole}] Gain macro (PAS - Std) = {gain_macro:+.4f}")

        resultats.append({
            "alpha": alpha,
            "cible": cible,
            "couv_marg_std": marg_std,
            "couv_marg_pas": marg_pas,
            "couv_macro_std": macro_std,
            "couv_macro_pas": macro_pas,
            "taille_std": taille_std,
            "taille_pas": taille_pas,
            "q_std": q_std,
            "q_pas": q_pas,
            "gain_macro": gain_macro,
        })

    return pd.DataFrame(resultats)


# ================================================================
# ETAPE 9 : VISUALISATIONS
# ================================================================
def generer_figures(df_res):
    print(f"\n{'=' * 65}")
    print("GENERATION DES FIGURES")
    print(f"{'=' * 65}")

    # --- Figure 1 : Couverture observee vs cible ---
    fig, ax = plt.subplots(figsize=(9, 7))
    cibles = df_res["cible"].values
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5,
            label="Couverture parfaite (y = x)")
    ax.scatter(cibles, df_res["couv_marg_std"], s=140, color="steelblue",
               edgecolor="black", marker="o", label="Standard CP", zorder=5)
    ax.scatter(cibles, df_res["couv_marg_pas"], s=140, color="tomato",
               edgecolor="black", marker="s", label="PAS", zorder=5)

    for i, alpha in enumerate(ALPHA_GRID):
        x = cibles[i]
        ax.annotate(f"alpha={alpha}",
                    (x, df_res["couv_marg_std"].iloc[i]),
                    textcoords="offset points", xytext=(8, 8), fontsize=9)

    ax.set_xlabel("Couverture cible (1 - alpha)", fontsize=12)
    ax.set_ylabel("Couverture observee", fontsize=12)
    ax.set_title("Validation : Couverture observee vs Couverture cible",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.75, 1.02)
    ax.set_ylim(0.75, 1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_FIG, "fig_couverture_vs_cible.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  fig_couverture_vs_cible.png OK")

    # --- Figure 2 : Couverture MACRO vs alpha ---
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(df_res["alpha"], df_res["couv_macro_std"], "o-",
            color="steelblue", linewidth=2, markersize=10,
            label="Standard CP")
    ax.plot(df_res["alpha"], df_res["couv_macro_pas"], "s-",
            color="tomato", linewidth=2, markersize=10,
            label="PAS")
    ax.plot(df_res["alpha"], df_res["cible"], "k--", alpha=0.5,
            label="Cible (1 - alpha)")

    ax.set_xlabel("alpha", fontsize=12)
    ax.set_ylabel("Couverture macro", fontsize=12)
    ax.set_title("Couverture macro : Standard vs PAS",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_FIG, "fig_couverture_macro_vs_alpha.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  fig_couverture_macro_vs_alpha.png OK")

    # --- Figure 3 : Taille vs alpha ---
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(df_res["alpha"], df_res["taille_std"], "o-", color="steelblue",
            linewidth=2, markersize=10, label="Standard CP")
    ax.plot(df_res["alpha"], df_res["taille_pas"], "s-", color="tomato",
            linewidth=2, markersize=10, label="PAS")
    ax.set_xlabel("alpha", fontsize=12)
    ax.set_ylabel("Taille moyenne des ensembles", fontsize=12)
    ax.set_title("Compromis couverture-efficacite",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_FIG, "fig_taille_vs_alpha.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  fig_taille_vs_alpha.png OK")

    # --- Figure 4 : Gain macro de PAS sur Standard ---
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["tomato" if g < 0 else "green" for g in df_res["gain_macro"]]
    bars = ax.bar(range(len(df_res)), df_res["gain_macro"],
                   color=colors, edgecolor="black")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(range(len(df_res)))
    ax.set_xticklabels([f"alpha={a}" for a in ALPHA_GRID])
    ax.set_ylabel("Gain de couverture macro (PAS - Standard)", fontsize=12)
    ax.set_title("PAS apporte-t-il un gain sur la couverture macro ?",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, df_res["gain_macro"]):
        offset = 0.001 if val >= 0 else -0.003
        ax.text(bar.get_x() + bar.get_width()/2, val + offset,
                f"{val:+.4f}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_FIG, "fig_gain_macro.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  fig_gain_macro.png OK")


# ================================================================
# ETAPE 10 : SAUVEGARDE
# ================================================================
def sauvegarder_resultats(df_res):
    out_csv = os.path.join(DIR_OUT, "resultats_multi_alpha.csv")
    df_res.to_csv(out_csv, index=False)
    print(f"\n  CSV sauvegarde : {out_csv}")

    print(f"\n{'=' * 65}")
    print("TABLEAU RECAPITULATIF")
    print(f"{'=' * 65}")
    print(df_res.to_string(index=False))


# ================================================================
# MAIN
# ================================================================
def main():
    calib, test, scores = charger_donnees()
    prevalence = calculer_prevalence(calib)
    df_std = calculer_scores_standard(calib, scores)
    df_pas = calculer_scores_pas(df_std, prevalence)
    df_res = boucle_alpha(df_std, df_pas, test, scores, prevalence)
    generer_figures(df_res)
    sauvegarder_resultats(df_res)

    print(f"\n{'=' * 65}")
    print("PHASE 1 TERMINEE")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrompu")
        sys.exit(130)
    except Exception as e:
        print(f"\nERREUR : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)