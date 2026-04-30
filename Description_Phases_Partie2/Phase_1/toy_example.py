"""
=============================================================================
Partie 2 — Phase 1 : Sanity Check sur Dataset Synthétique
=============================================================================
Objectif : Valider notre implémentation de la Conformal Prediction
           en la testant sur un dataset où on contrôle tout.
           Si on obtient ~95% de couverture marginale, le code est validé.

Données : Mélange de 10 gaussiennes en 2D (X1, X2)
Modèle  : Régression Logistique Multinomiale
Split   : Bloc A (5000, train) / D_cal (2500) / D_test (2500)

Demandé par : Joseph Salmon (mail du 21/04/2026)
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

np.random.seed(42)

# ================================================================
# PARAMÈTRES
# ================================================================
N = 10_000          # Nombre total d'observations
K = 10              # Nombre de classes
alpha = 0.05        # Niveau de risque (couverture cible = 95%)

# ================================================================
# ÉTAPE 1 : Génération d'un mélange de gaussiennes
# ================================================================
print("=" * 65)
print("PARTIE 2 — PHASE 1 : SANITY CHECK SUR DONNÉES SYNTHÉTIQUES")
print("=" * 65)

# Centres des 10 classes disposés en cercle (bien séparés en 2D)
angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
rayon = 4.0
centres = np.column_stack([rayon * np.cos(angles), rayon * np.sin(angles)])

# Générer 1000 points par classe (total = 10 000)
n_par_classe = N // K
X_list, y_list = [], []

for k in range(K):
    X_k = np.random.randn(n_par_classe, 2) + centres[k]
    y_k = np.full(n_par_classe, k)
    X_list.append(X_k)
    y_list.append(y_k)

X = np.vstack(X_list)
y = np.concatenate(y_list)

# Mélanger les données
idx_shuffle = np.random.permutation(len(y))
X, y = X[idx_shuffle], y[idx_shuffle]

print(f"\nDonnées générées : mélange de {K} gaussiennes en 2D")
print(f"  N = {N} observations")
print(f"  K = {K} classes")
print(f"  p = 2 variables (X1, X2)")
print(f"  {n_par_classe} observations par classe")

# ================================================================
# ÉTAPE 2 : Split Bloc A (train) / D_cal / D_test
# ================================================================
X_blocA, X_blocB, y_blocA, y_blocB = train_test_split(
    X, y, test_size=0.5, random_state=42, stratify=y
)
X_cal, X_test, y_cal, y_test = train_test_split(
    X_blocB, y_blocB, test_size=0.5, random_state=42, stratify=y_blocB
)

print(f"\nSplit des données :")
print(f"  Bloc A (entraînement) : {len(y_blocA)}")
print(f"  D_calibration         : {len(y_cal)}")
print(f"  D_test                : {len(y_test)}")

# ================================================================
# ÉTAPE 3 : Entraînement du classifieur
# ================================================================
print(f"\nEntraînement (Régression Logistique Multinomiale)...")
clf = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=1000,
    random_state=42
)
clf.fit(X_blocA, y_blocA)

acc_test = clf.score(X_test, y_test)
print(f"  Accuracy sur D_test : {acc_test:.3f}")

# Probabilités softmax
proba_cal = clf.predict_proba(X_cal)
proba_test = clf.predict_proba(X_test)
classes = clf.classes_

# ================================================================
# ÉTAPE 4 : CP Marginale (Baseline)
# ================================================================
print(f"\n{'=' * 65}")
print("CP MARGINALE (BASELINE)")
print(f"{'=' * 65}")

# Scores de non-conformité
scores_cal = np.array([
    1 - proba_cal[i, np.where(classes == y_cal[i])[0][0]]
    for i in range(len(y_cal))
])

# Quantile corrigé
n_cal = len(scores_cal)
niveau = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
niveau = min(niveau, 1.0)
q_hat = np.quantile(scores_cal, niveau, method="higher")

print(f"  n_calibration      = {n_cal}")
print(f"  Niveau du quantile = {niveau:.6f}")
print(f"  q̂                  = {q_hat:.6f}")
print(f"  Seuil de score     = {1 - q_hat:.6f}")

# Ensembles de prédiction
seuil_bl = 1 - q_hat
resultats_bl = []

for i in range(len(y_test)):
    ensemble = [classes[j] for j in range(K) if proba_test[i, j] >= seuil_bl]
    if len(ensemble) == 0:
        ensemble = [classes[np.argmax(proba_test[i])]]
    resultats_bl.append({
        "vraie_classe": y_test[i],
        "taille": len(ensemble),
        "couvert": y_test[i] in ensemble
    })

df_bl = pd.DataFrame(resultats_bl)

# Métriques
couv_marginale_bl = df_bl["couvert"].mean()
taille_moy_bl = df_bl["taille"].mean()

# Couverture MACRO
couv_par_classe_bl = {}
for k in range(K):
    sub = df_bl[df_bl["vraie_classe"] == k]
    if len(sub) > 0:
        couv_par_classe_bl[k] = sub["couvert"].mean()
couv_macro_bl = np.mean(list(couv_par_classe_bl.values()))

print(f"\n  Couverture MARGINALE : {couv_marginale_bl:.4f}  (cible : {1-alpha:.2f})")
print(f"  Couverture MACRO     : {couv_macro_bl:.4f}")
print(f"  Taille moyenne       : {taille_moy_bl:.2f}")

# ================================================================
# ÉTAPE 5 : Vérification
# ================================================================
print(f"\n{'=' * 65}")
print("VÉRIFICATION")
print(f"{'=' * 65}")

if 0.93 <= couv_marginale_bl <= 0.97:
    print(f"✅ Couverture marginale = {couv_marginale_bl:.4f} ∈ [0.93, 0.97]")
    print(f"   → Implémentation de la CP marginale VALIDÉE")
else:
    print(f"❌ Couverture marginale = {couv_marginale_bl:.4f} hors [0.93, 0.97]")
    print(f"   → Implémentation à VÉRIFIER")

# Détail par classe
print(f"\n  --- Couverture par classe ---")
print(f"  {'Classe':<10} {'n_test':>8} {'Couverture':>12}")
print(f"  {'-' * 30}")
for k in range(K):
    n_k = (y_test == k).sum()
    c_k = couv_par_classe_bl.get(k, 0)
    indicateur = "✅" if c_k >= 0.90 else "⚠"
    print(f"  {k:<10} {n_k:>8} {c_k:>12.4f}  {indicateur}")

# ================================================================
# ÉTAPE 6 : Visualisations
# ================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Figure 1 : Nuage de points 2D
scatter = axes[0].scatter(X_test[:, 0], X_test[:, 1], c=y_test,
                          cmap="tab10", alpha=0.5, s=10, edgecolors="none")
# Ajouter les centres
axes[0].scatter(centres[:, 0], centres[:, 1], c="black", marker="X",
                s=100, zorder=5, label="Centres")
axes[0].set_xlabel("$X_1$")
axes[0].set_ylabel("$X_2$")
axes[0].set_title(f"Mélange de {K} gaussiennes en 2D")
axes[0].legend()
plt.colorbar(scatter, ax=axes[0], label="Classe")

# Figure 2 : Couverture par classe
x_pos = np.arange(K)
axes[1].bar(x_pos, [couv_par_classe_bl[k] for k in range(K)],
            color="steelblue", edgecolor="black", alpha=0.7)
axes[1].axhline(y=1-alpha, color="green", linestyle="--", linewidth=1.5,
                label=f"Cible = {1-alpha:.0%}")
axes[1].axhline(y=couv_marginale_bl, color="red", linestyle=":",
                linewidth=1.5, label=f"Marginale = {couv_marginale_bl:.3f}")
axes[1].set_xlabel("Classe")
axes[1].set_ylabel("Couverture")
axes[1].set_title("Couverture par classe")
axes[1].set_xticks(x_pos)
axes[1].legend(fontsize=8)
axes[1].set_ylim(0.80, 1.05)

# Figure 3 : Distribution des tailles
taille_max = df_bl["taille"].max()
bins = range(1, taille_max + 2)
axes[2].hist(df_bl["taille"], bins=bins, color="steelblue",
             edgecolor="black", alpha=0.7, align="left")
axes[2].set_xlabel("Taille de l'ensemble C(x)")
axes[2].set_ylabel("Nombre d'observations")
axes[2].set_title(f"Taille des ensembles (moy = {taille_moy_bl:.2f})")

fig.suptitle(f"Sanity Check — CP Marginale sur mélange de gaussiennes\n"
             f"Couverture marginale = {couv_marginale_bl:.4f}, "
             f"Couverture macro = {couv_macro_bl:.4f}",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("fig_sanity_check.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n✓ fig_sanity_check.png")

# Distribution des scores de non-conformité
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(scores_cal, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
ax.axvline(q_hat, color="red", linestyle="--", linewidth=2,
           label=f"$\\hat{{q}}$ = {q_hat:.4f}")
ax.set_xlabel("Score de non-conformité $s_i = 1 - \\hat{p}_{y_i}$")
ax.set_ylabel("Fréquence")
ax.set_title("Distribution des scores de non-conformité (calibration)")
ax.legend()
plt.tight_layout()
plt.savefig("fig_scores_toy.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ fig_scores_toy.png")

print(f"\n{'=' * 65}")
print("SANITY CHECK TERMINÉ")
print(f"{'=' * 65}")
