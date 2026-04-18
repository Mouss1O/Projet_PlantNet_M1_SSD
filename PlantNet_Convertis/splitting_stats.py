"""
splitting_stats.py
Stats descriptives sur le split cal/test
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

BASE     = r"C:\Users\SCD-UM\Documents\Projet_PlantNet_M1_SSD"
DIR_DATA = os.path.join(BASE, "data", "split_random")
DIR_FIG  = os.path.join(BASE, "figures")

PATH_CAL  = os.path.join(DIR_DATA, "calibration.csv")
PATH_TEST = os.path.join(DIR_DATA, "test.csv")


def main():
    print("="*55)
    print("STATS DESCRIPTIVES DU SPLIT")
    print("="*55)
    
    for p in [PATH_CAL, PATH_TEST]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Manquant : {p}")
    
    os.makedirs(DIR_FIG, exist_ok=True)
    
    cal  = pd.read_csv(PATH_CAL)
    test = pd.read_csv(PATH_TEST)
    
    c_cal  = cal["ground_truth_val"].value_counts()
    c_test = test["ground_truth_val"].value_counts()
    
    sp_cal  = set(c_cal.index)
    sp_test = set(c_test.index)
    orphans = sp_test - sp_cal
    common  = sp_cal & sp_test
    only_cal = sp_cal - sp_test
    
    print(f"\n=== Repartition ===")
    print(f"Cal  : {len(cal):>6,} obs / {len(c_cal):,} especes")
    print(f"Test : {len(test):>6,} obs / {len(c_test):,} especes")
    print(f"\nEspeces communes      : {len(common):,}")
    print(f"Especes uniquement cal : {len(only_cal):,}")
    print(f"Especes uniquement test: {len(orphans):,} (orphelines)")
    
    # Observations concernees par les orphelines
    n_obs_orphan = test[test["ground_truth_val"].isin(orphans)].shape[0]
    print(f"\nObs de test orphelines : {n_obs_orphan:,} ({100*n_obs_orphan/len(test):.1f}%)")
    
    # Figure 1 : Distribution comparative par strate
    strates = [(1, 1), (2, 4), (5, 9), (10, 49), (50, float('inf'))]
    labels  = ['1', '2-4', '5-9', '10-49', '50+']
    
    cal_counts  = [((c_cal >= lo) & (c_cal <= hi)).sum() for lo, hi in strates]
    test_counts = [((c_test >= lo) & (c_test <= hi)).sum() for lo, hi in strates]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    width = 0.35
    b1 = ax.bar(x - width/2, cal_counts,  width, label='Calibration', color='steelblue')
    b2 = ax.bar(x + width/2, test_counts, width, label='Test',        color='coral')
    ax.set_xlabel("Observations par espece")
    ax.set_ylabel("Nombre d'especes")
    ax.set_title("Distribution des especes par strate de frequence")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    for bars in [b1, b2]:
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+5,
                    f'{int(b.get_height())}', ha='center', fontsize=8)
    plt.savefig(os.path.join(DIR_FIG, "07_split_strates.png"), dpi=100, bbox_inches='tight')
    plt.close()
    print("  07_split_strates.png OK")
    
    # Figure 2 : Orphelines (critique pour justifier PAS)
    fig, ax = plt.subplots(figsize=(8, 5))
    cats = ['Communes\n(cal et test)', 'Cal seulement', 'Test seulement\n(ORPHELINES)']
    vals = [len(common), len(only_cal), len(orphans)]
    colors = ['#2ca02c', '#1f77b4', '#d62728']
    bars = ax.bar(cats, vals, color=colors)
    ax.set_ylabel("Nombre d'especes")
    ax.set_title("Especes orphelines - Probleme long tail")
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+5,
                f'{v:,}', ha='center', fontsize=10, fontweight='bold')
    plt.savefig(os.path.join(DIR_FIG, "08_orphan_species.png"), dpi=100, bbox_inches='tight')
    plt.close()
    print("  08_orphan_species.png OK")
    
    # Figure 3 : Rang/frequence superpose cal vs test
    fig, ax = plt.subplots(figsize=(10, 6))
    cal_sorted  = c_cal.sort_values(ascending=False).values
    test_sorted = c_test.sort_values(ascending=False).values
    ax.loglog(np.arange(1, len(cal_sorted)+1),  cal_sorted,  'b-', label='Calibration', linewidth=1.5)
    ax.loglog(np.arange(1, len(test_sorted)+1), test_sorted, 'r-', label='Test',        linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Rang espece (log)")
    ax.set_ylabel("Observations (log)")
    ax.set_title("Long tail - Cal vs Test")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(DIR_FIG, "09_longtail_cal_test.png"), dpi=100, bbox_inches='tight')
    plt.close()
    print("  09_longtail_cal_test.png OK")
    
    # Export CSV
    summary = pd.DataFrame({
        "metric": ["n_obs_cal", "n_obs_test", "n_sp_cal", "n_sp_test",
                   "sp_communes", "sp_cal_only", "sp_orphelines",
                   "obs_orphelines", "pct_obs_orphelines"],
        "value":  [len(cal), len(test), len(c_cal), len(c_test),
                   len(common), len(only_cal), len(orphans),
                   n_obs_orphan, round(100*n_obs_orphan/len(test), 2)]
    })
    out_csv = os.path.join(BASE, "stats_summary_split.csv")
    summary.to_csv(out_csv, index=False)
    print(f"\nResume : {out_csv}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERREUR : {e}")
        sys.exit(1)