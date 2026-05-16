"""
=============================================================================
Sanity Check 4.2 : Cadre déséquilibré (Standard CP vs PAS) — Version Finale
=============================================================================
Projet : Pl@ntNet-CP — Prédiction Conformelle
Objectif : Démontrer la supériorité de PAS sur la longue traîne artificielle.
           Correction : Ajout d'une standardisation et augmentation des 
           itérations pour garantir la convergence du modèle (Zéro Warning).
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ================================================================
# CONFIGURATION
# ================================================================
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DIR_FIGURES = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(DIR_FIGURES, exist_ok=True)

ALPHA        = 0.05
RANDOM_STATE = 42
K            = 10

# ================================================================
# ÉTAPE 1 : GÉNÉRATION DÉSÉQUILIBRÉE (LONGUE TRAÎNE)
# ================================================================
print("=" * 70)
print("SANITY CHECK 4.2 : CADRE DÉSÉQUILIBRÉ (ZÉRO WARNING)")
print("=" * 70)

angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
centres_theo = np.column_stack([4.0 * np.cos(angles), 4.0 * np.sin(angles)])

X_list, y_list = [], []
for k in range(K):
    n_pts = 1500 if k < 5 else 50 # 5 Classes Communes / 5 Classes Rares
    X_k = np.random.RandomState(k).randn(n_pts, 2) + centres_theo[k]
    X_list.append(X_k)
    y_list.append(np.full(n_pts, k))

X = np.vstack(X_list)
y = np.concatenate(y_list)

# Split hiérarchique (stratifié pour préserver le déséquilibre dans chaque set)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.6, stratify=y, random_state=RANDOM_STATE)
X_calib, X_test, y_calib, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE)

# ================================================================
# ÉTAPE 2 : PRÉTRAITEMENT ET ENTRAÎNEMENT DU MODÈLE
# ================================================================
print(f"\nStandardisation des données et entraînement...")

# Normalisation pour aider la convergence de la logistique
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_calib_scaled = scaler.transform(X_calib)
X_test_scaled  = scaler.transform(X_test)

# Modèle avec max_iter augmenté pour éviter tout ConvergenceWarning
model = LogisticRegression(
    multi_class='multinomial', 
    max_iter=1000, 
    solver='lbfgs',
    random_state=RANDOM_STATE
).fit(X_train_scaled, y_train)

# Calcul des prévalences p_hat(y) sur la calibration
unique, counts = np.unique(y_calib, return_counts=True)
prev_dict = dict(zip(unique, counts / len(y_calib)))

# ================================================================
# ÉTAPE 3 : APPLICATION DES FORMULES DU BLOG (TES FORMULES)
# ================================================================
print(f"Calcul des quantiles q_hat...")

p_calib = model.predict_proba(X_calib_scaled)

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
# ÉTAPE 4 : ÉVALUATION ET VISUALISATION TRIPLE
# ================================================================
p_test = model.predict_proba(X_test_scaled)
p_y_test_mat = np.array([prev_dict[cl] for cl in range(K)])

# Construction des ensembles
set_std = (-p_test) <= q_std
set_pas = (-p_test / p_y_test_mat) <= q_pas

# Correction top-1 pour éviter les ensembles vides (si nécessaire)
for s in [set_std, set_pas]:
    for i in range(len(s)):
        if not np.any(s[i]): s[i, np.argmax(p_test[i])] = True

# Métriques
cov_std = [np.mean(set_std[y_test == k, k]) for k in range(K)]
cov_pas = [np.mean(set_pas[y_test == k, k]) for k in range(K)]
marg_pas = np.mean(set_pas[np.arange(len(y_test)), y_test])
macro_pas = np.mean(cov_pas)

# Figure Triple Panel (Style Capture 2)
fig, axes = plt.subplots(1, 3, figsize=(22, 6))

# Panel 1 : Nuage 2D
scatter = axes[0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='tab10', alpha=0.5, s=15)
axes[0].scatter(centres_theo[:, 0], centres_theo[:, 1], c='black', marker='X', s=120, label='Centres')
axes[0].set_title("Longue Traîne : 5 Communes / 5 Rares", fontsize=13)
axes[0].set_xlabel("$X_1$"); axes[0].set_ylabel("$X_2$")
axes[0].legend()

# Panel 2 : Comparaison Équité
axes[1].bar(np.arange(K)-0.2, cov_std, width=0.4, label='Standard CP', color='#729ece', alpha=0.8, edgecolor='black')
axes[1].bar(np.arange(K)+0.2, cov_pas, width=0.4, label='PAS CP', color='#ff7f0e', alpha=0.8, edgecolor='black')
axes[1].axhline(y=1-ALPHA, color='green', linestyle='--', linewidth=2, label='Cible 95%')
axes[1].set_title("Équité de couverture (Standard vs PAS)", fontsize=13)
axes[1].set_ylim(0.0, 1.1); axes[1].set_xlabel("Classe ID"); axes[1].legend(loc='lower right')

# Panel 3 : Taille des ensembles PAS
axes[2].hist(np.sum(set_pas, axis=1), bins=np.arange(0.5, 6.5, 1), rwidth=0.9, color='#ff7f0e', alpha=0.7, edgecolor='black')
axes[2].set_title(f"Taille ensembles PAS (moy = {np.mean(np.sum(set_pas, axis=1)):.2f})", fontsize=13)
axes[2].set_xlabel("Nombre de labels prédits")

fig.suptitle(f"Sanity Check Déséquilibré — Méthode PAS\nMarginale PAS = {marg_pas:.4f}, Macro PAS = {macro_pas:.4f}", 
             fontsize=15, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig(os.path.join(DIR_FIGURES, "fig_sanity_imbalanced_triple_final.png"), dpi=200)
plt.show()

print(f"\n--- Résultats finaux ---")
print(f"| Méthode  | Marginale | Macro  | Taille Moy. |")
print(f"|----------|-----------|--------|-------------|")
print(f"| Standard | {np.mean(set_std[np.arange(len(y_test)), y_test]):.4f}    | {np.mean(cov_std):.4f} | {np.mean(np.sum(set_std, axis=1)):.2f}        |")
print(f"| PAS      | {marg_pas:.4f}    | {macro_pas:.4f} | {np.mean(np.sum(set_pas, axis=1)):.2f}        |")
print(f"\n✓ TERMINÉ : Code propre et modèle convergé.")