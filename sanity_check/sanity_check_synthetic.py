"""
sanity_check_synthetic.py - Phase 1 
Validation de l'implementation Conformal Prediction sur donnees synthetiques.

Objectif : verifier que notre code atteint la couverture cible (95%)
sur un dataset controle. Si OK, l'implementation est validee.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ================================================================
# CHEMINS RELATIFS
# ================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.dirname(SCRIPT_DIR)
DIR_FIG    = os.path.join(ROOT, "figures_sanity")
os.makedirs(DIR_FIG, exist_ok=True)

# ================================================================
# PARAMETRES
# ================================================================
N_TOTAL        = 10_000
N_CLASSES      = 10
N_FEATURES     = 4
ALPHA          = 0.05         # niveau d'erreur (couverture cible = 95%)
SEED           = 42
N_REPLICATIONS = 10           # pour mesurer la stabilite


def conformal_calibrate(scores_cal: np.ndarray, alpha: float) -> float:
    """Quantile conformal avec correction finie-sample."""
    n     = len(scores_cal)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    q_hat = float(np.quantile(scores_cal, level, method='higher'))
    return q_hat


def evaluate_coverage(proba_test: np.ndarray,
                       y_test: np.ndarray,
                       q_hat: float):
    """
    Construit les ensembles de prediction et evalue.
    
    Returns
    -------
    coverage   : taux de couverture observe
    set_sizes  : taille de l'ensemble pour chaque obs
    covered    : booleen, vraie classe dans ensemble
    """
    n = len(y_test)
    threshold = 1.0 - q_hat
    
    in_set    = proba_test >= threshold
    set_sizes = in_set.sum(axis=1)
    
    covered  = in_set[np.arange(n), y_test]
    coverage = covered.mean()
    
    return coverage, set_sizes, covered


def run_one_replication(seed: int, alpha: float = ALPHA):
    """Execute une replication complete du pipeline."""
    
    # 1. Generer donnees
    X, y = make_classification(
        n_samples=N_TOTAL,
        n_features=N_FEATURES,
        n_informative=N_FEATURES,
        n_redundant=0,
        n_classes=N_CLASSES,
        n_clusters_per_class=1,
        random_state=seed
    )
    
    # 2. Split bloc A / bloc B (50/50)
    X_A, X_B, y_A, y_B = train_test_split(
        X, y, test_size=0.5, random_state=seed
    )
    
    # 3. Entrainer sur bloc A
    # le multinomial est le comportement par defaut avec solver='lbfgs'
    model = LogisticRegression(
        max_iter=1000,
        random_state=seed
    )
    model.fit(X_A, y_A)
    
    # 4. Split bloc B en cal / test (50/50)
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_B, y_B, test_size=0.5, random_state=seed
    )
    
    # 5. Probabilites predites
    proba_cal  = model.predict_proba(X_cal)
    proba_test = model.predict_proba(X_test)
    
    # 6. Scores de non-conformite : s = 1 - p(vraie classe)
    n_cal      = len(y_cal)
    p_true_cal = proba_cal[np.arange(n_cal), y_cal]
    scores_cal = 1.0 - p_true_cal
    
    # 7. Calibration : q_hat
    q_hat = conformal_calibrate(scores_cal, alpha)
    
    # 8. Evaluation sur test
    coverage, set_sizes, covered = evaluate_coverage(
        proba_test, y_test, q_hat
    )
    
    # 9. Accuracy du modele (sanity check additionnel)
    accuracy = model.score(X_test, y_test)
    
    return {
        "seed"        : seed,
        "q_hat"       : q_hat,
        "coverage"    : coverage,
        "avg_size"    : set_sizes.mean(),
        "median_size" : np.median(set_sizes),
        "max_size"    : set_sizes.max(),
        "min_size"    : set_sizes.min(),
        "size_1_pct"  : (set_sizes == 1).mean(),
        "model_acc"   : accuracy,
        "set_sizes"   : set_sizes,
        "covered"     : covered,
        "y_test"      : y_test,
    }


def main():
    print("="*60)
    print("SANITY CHECK - CONFORMAL PREDICTION SUR DONNEES SYNTHETIQUES")
    print("="*60)
    print(f"N total       : {N_TOTAL}")
    print(f"N classes     : {N_CLASSES}")
    print(f"N features    : {N_FEATURES}")
    print(f"Alpha         : {ALPHA}")
    print(f"Cible         : {1-ALPHA:.0%}")
    print(f"Replications  : {N_REPLICATIONS}")
    print(f"Bornes accept.: 94% - 96%")
    
    # Replications pour stabilite
    results = []
    for i in range(N_REPLICATIONS):
        seed_i = SEED + i
        print(f"\n--- Replication {i+1}/{N_REPLICATIONS} (seed={seed_i}) ---")
        try:
            res = run_one_replication(seed_i)
            results.append(res)
            print(f"  Accuracy modele : {res['model_acc']:.4f}")
            print(f"  q_hat           : {res['q_hat']:.4f}")
            print(f"  Coverage        : {res['coverage']:.4f}  ({res['coverage']*100:.2f}%)")
            print(f"  Taille moyenne  : {res['avg_size']:.2f}")
        except Exception as e:
            print(f"  ERREUR : {e}")
            continue
    
    if not results:
        raise RuntimeError("Aucune replication n'a reussi")
    
    # Synthese
    coverages = np.array([r["coverage"] for r in results])
    sizes_avg = np.array([r["avg_size"] for r in results])
    accs      = np.array([r["model_acc"] for r in results])
    
    print(f"\n{'='*60}")
    print("RESULTATS GLOBAUX")
    print(f"{'='*60}")
    print(f"Couverture (cible = {1-ALPHA:.0%}) :")
    print(f"  Moyenne  : {coverages.mean():.4f}  ({coverages.mean()*100:.2f}%)")
    print(f"  Std      : {coverages.std():.4f}")
    print(f"  Min      : {coverages.min():.4f}")
    print(f"  Max      : {coverages.max():.4f}")
    
    in_target_zone = ((coverages >= 0.94) & (coverages <= 0.96)).sum()
    print(f"\n  Replications dans [94%, 96%] : {in_target_zone}/{N_REPLICATIONS}")
    
    if in_target_zone >= N_REPLICATIONS - 1:
        print(f"\n  [OK] IMPLEMENTATION VALIDEE")
    else:
        print(f"\n  [ATTENTION] Validation partielle - investiguer")
    
    print(f"\nTaille moyenne des ensembles :")
    print(f"  Moyenne  : {sizes_avg.mean():.2f}")
    print(f"  Std      : {sizes_avg.std():.2f}")
    
    print(f"\nAccuracy du modele (sanity additionnel) :")
    print(f"  Moyenne  : {accs.mean():.4f}")
    
    # Sauvegarde du tableau de resultats
    summary_df = pd.DataFrame([{
        k: v for k, v in r.items() 
        if k not in ["set_sizes", "covered", "y_test"]
    } for r in results])
    
    out_csv = os.path.join(ROOT, "sanity_check_results.csv")
    summary_df.to_csv(out_csv, index=False)
    print(f"\nResultats CSV : {out_csv}")
    
    # ============================================================
    # VISUALISATIONS
    # ============================================================
    
    # Figure 1 : Distribution des coverages sur les replications
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(coverages, bins=15, color='steelblue', edgecolor='white', alpha=0.7)
    ax.axvline(1-ALPHA, color='red', linestyle='--', linewidth=2,
               label=f'Cible = {1-ALPHA:.0%}')
    ax.axvline(coverages.mean(), color='green', linestyle='-', linewidth=2,
               label=f'Moyenne = {coverages.mean():.4f}')
    ax.axvspan(0.94, 0.96, alpha=0.15, color='green',
               label='Zone acceptable [94%, 96%]')
    ax.set_xlabel("Coverage observee")
    ax.set_ylabel("Nombre de replications")
    ax.set_title(f"Distribution des coverages sur {N_REPLICATIONS} replications")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(DIR_FIG, "01_coverage_distribution.png"),
                dpi=120, bbox_inches='tight')
    plt.close()
    print("  01_coverage_distribution.png OK")
    
    # Figure 2 : Coverage par replication
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(1, len(coverages)+1)
    colors = ['green' if 0.94 <= c <= 0.96 else 'orange' for c in coverages]
    ax.bar(x, coverages, color=colors, edgecolor='white')
    ax.axhline(1-ALPHA, color='red', linestyle='--', linewidth=2,
               label=f'Cible {1-ALPHA:.0%}')
    ax.axhline(0.94, color='gray', linestyle=':', alpha=0.7)
    ax.axhline(0.96, color='gray', linestyle=':', alpha=0.7)
    ax.set_xlabel("Replication")
    ax.set_ylabel("Coverage")
    ax.set_title("Coverage par replication")
    ax.set_xticks(x)
    ax.set_ylim(0.90, 1.00)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(DIR_FIG, "02_coverage_per_rep.png"),
                dpi=120, bbox_inches='tight')
    plt.close()
    print("  02_coverage_per_rep.png OK")
    
    # Figure 3 : Distribution des tailles d'ensemble (replication 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    sizes_first = results[0]["set_sizes"]
    unique, counts = np.unique(sizes_first, return_counts=True)
    bars = ax.bar(unique, counts, color='steelblue', edgecolor='white')
    ax.set_xlabel("Taille de l'ensemble C(x)")
    ax.set_ylabel("Nombre d'observations")
    ax.set_title(f"Distribution des tailles d'ensemble (replication 1)\n"
                 f"Moyenne = {sizes_first.mean():.2f} | "
                 f"Mediane = {np.median(sizes_first):.0f}")
    for b, c in zip(bars, counts):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+5,
                f'{int(c)}', ha='center', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.savefig(os.path.join(DIR_FIG, "03_set_sizes.png"),
                dpi=120, bbox_inches='tight')
    plt.close()
    print("  03_set_sizes.png OK")
    
    # Figure 4 : Visualisation 2D (projection sur 2 premieres features)
    np.random.seed(SEED)
    X, y = make_classification(
        n_samples=N_TOTAL, n_features=N_FEATURES,
        n_informative=N_FEATURES, n_redundant=0,
        n_classes=N_CLASSES, n_clusters_per_class=1,
        random_state=SEED
    )
    X_A, X_B, y_A, y_B = train_test_split(X, y, test_size=0.5, random_state=SEED)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(X_A[:, 0], X_A[:, 1], c=y_A,
                          cmap='tab10', alpha=0.5, s=10)
    ax.set_xlabel("X1 (1ere feature)")
    ax.set_ylabel("X2 (2eme feature)")
    ax.set_title(f"Donnees synthetiques (projection sur 2 features sur {N_FEATURES})\n"
                 f"{N_TOTAL//2} obs entrainement, {N_CLASSES} classes")
    plt.colorbar(scatter, label='Classe')
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(DIR_FIG, "04_data_2d.png"),
                dpi=120, bbox_inches='tight')
    plt.close()
    print("  04_data_2d.png OK")
    
    print(f"\n{'='*60}")
    print("TERMINE")
    print(f"{'='*60}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrompu")
        sys.exit(130)
    except Exception as e:
        print(f"\nERREUR : {e}")
        sys.exit(1)