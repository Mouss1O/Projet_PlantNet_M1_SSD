"""
=============================================================================
Sanity Check 4.1 : Cadre équilibré (Standard CP vs PAS)
=============================================================================
Projet : Pl@ntNet-CP — Prédiction Conformelle
Objectif : Valider la cohérence des deux méthodes sur un dataset équilibré.
           Vérifier que PAS converge vers Standard CP quand la prévalence est uniforme.
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ================================================================
# CONFIGURATION
# ================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_FIGURES = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(DIR_FIGURES, exist_ok=True)

ALPHA = 0.05
RANDOM_STATE = 42
N_SAMPLES = 5000
K = 10

# ================================================================
# ÉTAPE 1 : GÉNÉRATION DES DONNÉES (ÉQUILIBRÉES)
# ================================================================
print("=" * 70)
print("SANITY CHECK 4.1 : CADRE ÉQUILIBRÉ")
print("=" * 70)

angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
centres_theo = np.column_stack([4.0 * np.cos(angles), 4.0 * np.sin(angles)])
X, y = make_blobs(n_samples=N_SAMPLES, centers=centres_theo, cluster_std=1.2, random_state=RANDOM_STATE)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, stratify=y, random_state=RANDOM_STATE)
X_calib, X_test, y_calib, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE)

model = LogisticRegression(multi_class='multinomial', max_iter=1000).fit(X_train, y_train)

# Calcul prévalence sur calibration
unique, counts = np.unique(y_calib, return_counts=True)
prev_dict = dict(zip(unique, counts / len(y_calib)))

# ================================================================
# ÉTAPE 3 : APPLICATION DES FORMULES DU BLOG (TES FORMULES)
# ================================================================
print(f"\nCalcul des quantiles q_hat...")

p_calib = model.predict_proba(X_calib)

# --- MÉTHODE 1 : STANDARD CP ---
s_std_calib = -p_calib[np.arange(len(y_calib)), y_calib.astype(int)]
q_std = np.quantile(s_std_calib, np.ceil((len(y_calib)+1)*(1-ALPHA))/len(y_calib))

# --- MÉTHODE 2 : PAS CP (Prevalence-Adjusted) ---
p_y_calib = np.array([prev_dict[label] for label in y_calib])
s_pas_calib = -(p_calib[np.arange(len(y_calib)), y_calib.astype(int)] / p_y_calib)
q_pas = np.quantile(s_pas_calib, np.ceil((len(y_calib)+1)*(1-ALPHA))/len(y_calib))

print(f"  q_std : {q_std:.4f}")
print(f"  q_pas : {q_pas:.4f}")

# ================================================================
# ÉVALUATION ET VISUALISATION
# ================================================================
p_test = model.predict_proba(X_test)
p_y_test_mat = np.array([prev_dict[cl] for cl in range(K)])

set_std = (-p_test) <= q_std
set_pas = (-p_test / p_y_test_mat) <= q_pas

# Correction top-1 pour éviter ensembles vides
for s in [set_std, set_pas]:
    for i in range(len(s)):
        if not np.any(s[i]): s[i, np.argmax(p_test[i])] = True

cov_std = [np.mean(set_std[y_test == k, k]) for k in range(K)]
cov_pas = [np.mean(set_pas[y_test == k, k]) for k in range(K)]
marg_std = np.mean(set_std[np.arange(len(y_test)), y_test])
macro_std = np.mean(cov_std)

fig, axes = plt.subplots(1, 3, figsize=(22, 6))
# 1. 2D Plot
scatter = axes[0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='tab10', alpha=0.5, s=15)
axes[0].scatter(centres_theo[:, 0], centres_theo[:, 1], c='black', marker='X', s=120, label='Centres')
axes[0].set_title("Mélange de 10 gaussiennes en 2D"); axes[0].legend()
plt.colorbar(scatter, ax=axes[0], label="Classe")

# 2. Barplot
axes[1].bar(np.arange(K)-0.2, cov_std, width=0.4, label='Standard CP', color='#729ece', alpha=0.8, edgecolor='black')
axes[1].bar(np.arange(K)+0.2, cov_pas, width=0.4, label='PAS CP', color='#ff7f0e', alpha=0.8, edgecolor='black')
axes[1].axhline(y=1-ALPHA, color='green', linestyle='--', linewidth=2, label='Cible 95%')
axes[1].set_title("Couverture par classe"); axes[1].set_ylim(0.8, 1.05); axes[1].legend(loc='lower left')

# 3. Histogramme
axes[2].hist(np.sum(set_std, axis=1), bins=np.arange(0.5, 4.5, 1), rwidth=0.9, color='#729ece', alpha=0.7, edgecolor='black')
axes[2].set_title(f"Taille des ensembles (moy = {np.mean(np.sum(set_std, axis=1)):.2f})")

fig.suptitle(f"Sanity Check Équilibré — CP Marginale sur mélange de gaussiennes\nCouverture marginale = {marg_std:.4f}, Couverture macro = {macro_std:.4f}", fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(DIR_FIGURES, "fig_sanity_balanced_triple.png"), dpi=200)
plt.show()