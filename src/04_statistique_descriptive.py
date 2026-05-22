"""
=============================================================================
Statistiques descriptives : Visualisation de la Longue Traîne
=============================================================================
Projet : Pl@ntNet-CP — Prédiction Conformelle
Objectif : Montrer la distribution de la prévalence des espèces
           dans le jeu de données expert (structure de loi de puissance).

Entrées :
    - Observations_experts.csv  (observation_id, true_species_id)

Sorties :
    - figures/fig_longue_traine.png
    - figures/fig_histogramme_prevalence.png
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ================================================================
# CONFIGURATION
# ================================================================
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DIR_FIGURES = os.path.join(SCRIPT_DIR, "figures")

PATH_EXPERTS = os.path.join(SCRIPT_DIR, "observations_experts.csv")

os.makedirs(DIR_FIGURES, exist_ok=True)

# ================================================================
# CHARGEMENT
# ================================================================
print("=" * 60)
print("STATISTIQUES DESCRIPTIVES : LONGUE TRAÎNE")
print("=" * 60)

experts = pd.read_csv(PATH_EXPERTS)
experts.columns = ['observation_id', 'ground_truth_val']
experts['observation_id'] = experts['observation_id'].astype(str)
experts['ground_truth_val'] = experts['ground_truth_val'].astype(str)

print(f"Observations expertes totales : {len(experts):,}")

# ================================================================
# CALCUL DE LA PRÉVALENCE (nombre d'images par espèce)
# ================================================================
prevalence = (
    experts
    .groupby("ground_truth_val")
    .size()
    .reset_index(name="n_images")
    .sort_values("n_images", ascending=False)
    .reset_index(drop=True)
)

# Rang (de la plus commune à la plus rare)
prevalence["rank"] = prevalence.index + 1

# ================================================================
# STATISTIQUES RÉSUMÉES
# ================================================================
n_especes = len(prevalence)
moy = prevalence["n_images"].mean()
med = prevalence["n_images"].median()
maxi = prevalence["n_images"].max()
n_1_obs = (prevalence["n_images"] == 1).sum()
n_lt5 = (prevalence["n_images"] < 5).sum()
n_lt10 = (prevalence["n_images"] < 10).sum()
n_gte50 = (prevalence["n_images"] >= 50).sum()

print(f"\n--- Résumé de la prévalence ---")
print(f"Espèces uniques        : {n_especes:,}")
print(f"Moyenne obs/espèce     : {moy:.1f}")
print(f"Médiane obs/espèce     : {med:.0f}")
print(f"Maximum obs/espèce     : {maxi:,}")
print(f"Espèces avec 1 obs     : {n_1_obs:,} ({100*n_1_obs/n_especes:.1f}%)")
print(f"Espèces avec < 5 obs   : {n_lt5:,} ({100*n_lt5/n_especes:.1f}%)")
print(f"Espèces avec < 10 obs  : {n_lt10:,} ({100*n_lt10/n_especes:.1f}%)")
print(f"Espèces avec >= 50 obs : {n_gte50:,} ({100*n_gte50/n_especes:.1f}%)")

# ================================================================
# FIGURE 1 : Courbe de la Longue Traîne (rang vs prévalence)
# ================================================================
fig, ax = plt.subplots(figsize=(10, 6))

ax.fill_between(prevalence["rank"], prevalence["n_images"],
                alpha=0.3, color="steelblue")
ax.plot(prevalence["rank"], prevalence["n_images"],
        color="steelblue", linewidth=1.5)

ax.axhline(y=10, color="red", linestyle="--", linewidth=1.2,
           label=f"Seuil de rareté (n = 10)")

ax.annotate(f"{n_lt10:,} espèces rares\n({100*n_lt10/n_especes:.0f}%)",
            xy=(n_especes * 0.6, 5), fontsize=11, color="red",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                      edgecolor="red", alpha=0.8))

ax.annotate(f"{n_especes - n_lt10:,} espèces communes\n({100*(n_especes-n_lt10)/n_especes:.0f}%)",
            xy=(50, maxi * 0.3), fontsize=11, color="steelblue",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                      edgecolor="steelblue", alpha=0.8))

ax.set_yscale("log")
ax.set_xlabel("Rang de l'espèce (de la plus commune à la plus rare)", fontsize=12)
ax.set_ylabel("Nombre d'observations (échelle log)", fontsize=12)
ax.set_title("Distribution Long-Tail de la prévalence des espèces\n"
             f"({n_especes:,} espèces, {len(experts):,} observations expertes)",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(DIR_FIGURES, "fig_longue_traine.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print(f"\n✓ Figure sauvegardée : fig_longue_traine.png")

# ================================================================
# FIGURE 2 : Histogramme de la prévalence
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(prevalence["n_images"], bins=50, color="steelblue",
             edgecolor="black", alpha=0.7)
axes[0].axvline(x=10, color="red", linestyle="--", linewidth=1.5,
                label="Seuil = 10")
axes[0].set_xlabel("Nombre d'observations par espèce")
axes[0].set_ylabel("Nombre d'espèces")
axes[0].set_title("Distribution complète")
axes[0].legend()

rares = prevalence[prevalence["n_images"] < 30]
axes[1].hist(rares["n_images"], bins=range(1, 31), color="tomato",
             edgecolor="black", alpha=0.7, align="left")
axes[1].axvline(x=10, color="red", linestyle="--", linewidth=1.5,
                label="Seuil = 10")
axes[1].set_xlabel("Nombre d'observations par espèce")
axes[1].set_ylabel("Nombre d'espèces")
axes[1].set_title("Zoom sur les espèces rares (< 30 obs)")
axes[1].legend()

fig.suptitle("Histogramme de la prévalence des espèces",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(DIR_FIGURES, "fig_histogramme_prevalence.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ Figure sauvegardée : fig_histogramme_prevalence.png")

# ================================================================
# TABLEAU RÉCAPITULATIF (pour le rapport)
# ================================================================
print(f"\n{'=' * 60}")
print("TABLEAU POUR LE RAPPORT")
print(f"{'=' * 60}")
print(f"| Statistique | Valeur |")
print(f"|---|---|")
print(f"| Observations expertes totales | {len(experts):,} |")
print(f"| Espèces uniques | {n_especes:,} |")
print(f"| Moyenne obs/espèce | {moy:.1f} |")
print(f"| Médiane obs/espèce | {med:.0f} |")
print(f"| Espèces rares (< 10 obs) | {n_lt10:,} ({100*n_lt10/n_especes:.1f}%) |")
print(f"| Espèces communes (≥ 10 obs) | {n_especes-n_lt10:,} ({100*(n_especes-n_lt10)/n_especes:.1f}%) |")
print(f"| Espèces avec 1 seule obs | {n_1_obs:,} ({100*n_1_obs/n_especes:.1f}%) |")
print(f"| Max obs pour une espèce | {maxi:,} |")

print(f"\n{'=' * 60}")
print("TERMINÉ")
print(f"{'=' * 60}")