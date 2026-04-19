"""
stat_desc.py
Stats descriptives sur ai_scores_all.csv
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

# ================================================================
# CHEMINS RELATIFS
# ================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.dirname(SCRIPT_DIR)

PATH_SCORES = os.path.join(ROOT, "Données", "ai_scores", "ai_scores_all.csv")
DIR_FIG     = os.path.join(ROOT, "figures")


def gini(x):
    x = np.sort(x.astype(float))
    n = len(x)
    return (2 * np.sum(np.arange(1, n+1) * x) / (n * x.sum())) - (n+1)/n


def main():
    print("="*55)
    print("EXPLORATION ai_scores_all.csv")
    print("="*55)
    print(f"Racine projet : {ROOT}")
    
    if not os.path.exists(PATH_SCORES):
        raise FileNotFoundError(f"Fichier manquant : {PATH_SCORES}")
    
    os.makedirs(DIR_FIG, exist_ok=True)
    
    size_mb = os.path.getsize(PATH_SCORES) / 1e6
    print(f"Taille fichier : {size_mb:.1f} MB")
    
    chunksize = 500_000
    
    n_rows_total      = 0
    obs_ids_seen      = set()
    species_row_count = {}
    obs_nscores       = {}
    obs_topscore      = {}
    
    print("\nLecture par chunks...")
    try:
        for i, chunk in enumerate(pd.read_csv(PATH_SCORES, chunksize=chunksize)):
            n_rows_total += len(chunk)
            obs_ids_seen.update(chunk["observation_id"].unique())
            
            vc = chunk["spicies_id"].value_counts()
            for sp, n in vc.items():
                species_row_count[sp] = species_row_count.get(sp, 0) + int(n)
            
            grp = chunk.groupby("observation_id")
            for obs_id, sub in grp:
                obs_nscores[obs_id] = obs_nscores.get(obs_id, 0) + len(sub)
                top_idx = sub["score"].idxmax()
                top_sc  = float(sub.loc[top_idx, "score"])
                top_sp  = int(sub.loc[top_idx, "spicies_id"])
                if obs_id not in obs_topscore or top_sc > obs_topscore[obs_id][1]:
                    obs_topscore[obs_id] = (top_sp, top_sc)
            
            if (i + 1) % 20 == 0:
                print(f"  ... {(i+1)*chunksize:>10,} lignes lues")
    
    except (pd.errors.EmptyDataError, pd.errors.ParserError, KeyError) as e:
        raise ValueError(f"Erreur lecture : {e}")
    
    print(f"\nTotal lignes         : {n_rows_total:,}")
    print(f"Total observations   : {len(obs_ids_seen):,}")
    print(f"Total especes        : {len(species_row_count):,}")
    
    nscores_arr = np.fromiter(obs_nscores.values(), dtype=np.int32)
    topsc_arr   = np.fromiter((v[1] for v in obs_topscore.values()), dtype=np.float32)
    
    del obs_nscores
    
    print("\n=== Scores par observation ===")
    print(f"  Moyenne   : {nscores_arr.mean():.1f}")
    print(f"  Mediane   : {int(np.median(nscores_arr))}")
    print(f"  Min / Max : {nscores_arr.min()} / {nscores_arr.max()}")
    
    print("\n=== Confiance top-1 ===")
    print(f"  Moyenne   : {topsc_arr.mean():.4f}")
    print(f"  Mediane   : {np.median(topsc_arr):.4f}")
    print(f"  < 0.5     : {(topsc_arr < 0.5).sum():,} obs ({100*(topsc_arr < 0.5).mean():.1f}%)")
    print(f"  > 0.9     : {(topsc_arr > 0.9).sum():,} obs ({100*(topsc_arr > 0.9).mean():.1f}%)")
    
    prevalence = pd.Series(species_row_count).sort_values(ascending=False)
    del species_row_count
    
    top1_counts = {}
    for obs_id, (sp, _) in obs_topscore.items():
        top1_counts[sp] = top1_counts.get(sp, 0) + 1
    del obs_topscore
    
    top1_series = pd.Series(top1_counts).sort_values(ascending=False)
    del top1_counts
    
    print("\n=== Long tail ===")
    print(f"  Especes totales     : {len(prevalence):,}")
    print(f"  Moyenne apparitions : {prevalence.mean():.1f}")
    print(f"  Mediane             : {int(prevalence.median())}")
    print(f"  Max                 : {prevalence.max():,}")
    
    print("\n=== Repartition par prevalence ===")
    for t in [1, 5, 10, 50, 100, 1000]:
        n   = (prevalence <= t).sum()
        pct = 100 * n / len(prevalence)
        print(f"  <= {t:4d} : {n:>6,} especes ({pct:5.1f}%)")
    
    cum_obs = prevalence.cumsum() / prevalence.sum() * 100
    print("\n=== Concentration ===")
    for pct_sp in [1, 5, 10, 25, 50]:
        idx = int(len(prevalence) * pct_sp / 100)
        print(f"  Top {pct_sp:2d}% especes = {cum_obs.iloc[idx-1]:.1f}% des predictions")
    
    gini_val = gini(prevalence.values)
    print(f"\n  Coefficient de Gini : {gini_val:.4f}")
    
    print("\nGeneration des figures...")
    
    def subsample(arr, max_n=500_000):
        if len(arr) > max_n:
            idx = np.random.choice(len(arr), max_n, replace=False)
            return arr[idx]
        return arr
    
    # Figure 1
    fig, ax = plt.subplots(figsize=(8, 5))
    ranks = np.arange(1, len(prevalence) + 1)
    ax.loglog(ranks, prevalence.values, 'b-', linewidth=1)
    ax.set_xlabel("Rang espece (log)")
    ax.set_ylabel("Apparitions (log)")
    ax.set_title("Long tail - Prevalence IA")
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(DIR_FIG, "01_long_tail.png"), dpi=100, bbox_inches='tight')
    plt.close(fig)
    print("  01_long_tail.png OK")
    
    # Figure 2
    fig, ax = plt.subplots(figsize=(8, 5))
    ranks_t = np.arange(1, len(top1_series) + 1)
    ax.loglog(ranks_t, top1_series.values, 'r-', linewidth=1)
    ax.set_xlabel("Rang espece (log)")
    ax.set_ylabel("Predictions top-1 (log)")
    ax.set_title("Distribution des predictions top-1")
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(DIR_FIG, "02_top1_distribution.png"), dpi=100, bbox_inches='tight')
    plt.close(fig)
    print("  02_top1_distribution.png OK")
    
    # Figure 3
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(subsample(nscores_arr), bins=50, color='steelblue', edgecolor='white')
    ax.axvline(np.median(nscores_arr), color='red', linestyle='--',
               label=f'Mediane = {int(np.median(nscores_arr))}')
    ax.set_xlabel("Nombre de scores (top-k)")
    ax.set_ylabel("Observations")
    ax.set_title("Taille du top-k par observation")
    ax.legend()
    plt.savefig(os.path.join(DIR_FIG, "03_topk_size.png"), dpi=100, bbox_inches='tight')
    plt.close(fig)
    print("  03_topk_size.png OK")
    
    # Figure 4
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(subsample(topsc_arr), bins=50, color='#2ca02c', edgecolor='white')
    ax.axvline(np.median(topsc_arr), color='red', linestyle='--',
               label=f'Mediane = {np.median(topsc_arr):.3f}')
    ax.set_xlabel("Score top-1")
    ax.set_ylabel("Observations")
    ax.set_title("Distribution confiance top-1")
    ax.legend()
    plt.savefig(os.path.join(DIR_FIG, "04_top1_confidence.png"), dpi=100, bbox_inches='tight')
    plt.close(fig)
    print("  04_top1_confidence.png OK")
    
    # Figure 5
    fig, ax = plt.subplots(figsize=(8, 5))
    cum_sp = np.arange(1, len(prevalence)+1) / len(prevalence) * 100
    ax.plot(cum_sp, cum_obs.values, 'b-', linewidth=2, label='Observe')
    ax.plot([0,100], [0,100], 'k--', alpha=0.5, label='Uniforme')
    ax.fill_between(cum_sp, cum_obs.values, cum_sp, alpha=0.15, color='red')
    ax.set_xlabel("% cumule especes")
    ax.set_ylabel("% cumule apparitions")
    ax.set_title(f"Courbe de Lorenz (Gini = {gini_val:.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(DIR_FIG, "05_lorenz.png"), dpi=100, bbox_inches='tight')
    plt.close(fig)
    print("  05_lorenz.png OK")
    
    # Figure 6
    fig, ax = plt.subplots(figsize=(8, 5))
    bins   = [1, 2, 6, 11, 51, 101, 1001, 10001, int(prevalence.max())+1]
    labels = ['1', '2-5', '6-10', '11-50', '51-100', '101-1k', '1k-10k', '10k+']
    hist, _ = np.histogram(prevalence.values, bins=bins)
    colors = ['#d62728','#ff7f0e','#ffbb78','#98df8a',
              '#2ca02c','#1f77b4','#17becf','#9467bd']
    bars = ax.bar(range(len(labels)), hist, color=colors, edgecolor='white')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0)
    ax.set_yscale('log')
    ax.set_xlabel("Nb apparitions")
    ax.set_ylabel("Especes (log)")
    ax.set_title("Repartition par prevalence")
    for b, v in zip(bars, hist):
        if v > 0:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()*1.15,
                    str(int(v)), ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_FIG, "06_prevalence_bins.png"), dpi=100, bbox_inches='tight')
    plt.close(fig)
    print("  06_prevalence_bins.png OK")
    
    summary = {
        "n_lignes_total"       : n_rows_total,
        "n_observations"       : len(obs_ids_seen),
        "n_especes_vues"       : len(prevalence),
        "topk_moyen"           : round(float(nscores_arr.mean()), 2),
        "topk_median"          : int(np.median(nscores_arr)),
        "score_top1_moyen"     : round(float(topsc_arr.mean()), 4),
        "score_top1_median"    : round(float(np.median(topsc_arr)), 4),
        "gini_prevalence"      : round(gini_val, 4),
        "especes_1_apparition" : int((prevalence == 1).sum()),
        "especes_lt_10"        : int((prevalence < 10).sum()),
    }
    
    out_csv = os.path.join(ROOT, "stats_summary_scores.csv")
    pd.DataFrame([summary]).to_csv(out_csv, index=False)
    print(f"\nResume CSV : {out_csv}")
    print(f"Figures    : {DIR_FIG}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrompu")
        sys.exit(130)
    except Exception as e:
        print(f"\nERREUR : {e}")
        sys.exit(1)