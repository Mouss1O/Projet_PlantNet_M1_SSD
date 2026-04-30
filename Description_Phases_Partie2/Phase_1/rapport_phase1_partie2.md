# Rapport Phase 1 : Sanity Check sur Dataset Synthétique

## Projet Pl@ntNet-CP — Prédiction Conformelle et Longue Traîne

**Auteur** : Dossou Agossou  
**Date** : Avril 2026  
**Encadrants** : Joseph Salmon, Christophe Botella, Jean-Baptiste Fermanian

---

## 1. Objectif

Valider mathématiquement notre implémentation de la Conformal Prediction en la testant sur un dataset **synthétique contrôlé**. Sur un problème où l'on maîtrise entièrement le processus génératif, la CP marginale **doit** atteindre une couverture de 95 % (± 2 %). Si c'est le cas, le code est validé. Sinon, il y a un bug à corriger.

Cette vérification a été demandée par M. Salmon (mail du 21/04/2026) : *"essayer un cadre de classification simplifié (un mélange de gaussiennes par exemple), où vous contrôlez le processus génératif"*.

---

## 2. Protocole expérimental

### 2.1 Génération des données

Nous avons créé un **mélange de 10 gaussiennes en 2D** :

| Paramètre | Valeur |
|---|---|
| Nombre d'observations $N$ | 10 000 |
| Nombre de classes $K$ | 10 |
| Nombre de variables $p$ | 2 ($X_1$, $X_2$) |
| Observations par classe | 1 000 (équilibré) |
| Distribution | Gaussienne, centres disposés en cercle (rayon = 4) |

Les 10 centres sont uniformément répartis sur un cercle de rayon 4 dans le plan ($X_1$, $X_2$), avec une variance unitaire par classe. Cette disposition crée des zones de chevauchement entre classes voisines, rendant le problème non trivial pour le classifieur.

### 2.2 Partitionnement des données

Le dataset est divisé en trois blocs :

| Ensemble | Taille | Rôle |
|---|---|---|
| Bloc A (entraînement) | 5 000 | Entraîner le classifieur |
| $D_{\text{calibration}}$ | 2 500 | Calculer le quantile conformal |
| $D_{\text{test}}$ | 2 500 | Évaluer la couverture |

Le split est stratifié (`stratify=y`) pour garantir la même proportion de classes dans chaque ensemble, avec `random_state=42` pour la reproductibilité.

### 2.3 Classifieur

Nous utilisons une **Régression Logistique Multinomiale** (implémentation `scikit-learn`) :

```
LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
```

L'accuracy obtenue sur $D_{\text{test}}$ est de **76,8 %**. Ce taux modéré est attendu : avec 10 classes et seulement 2 variables, les zones de chevauchement sont importantes. C'est un cadre réaliste pour tester la CP, car la méthode doit justement fonctionner même quand le classifieur est imparfait.

---

## 3. Implémentation de la CP Marginale

### 3.1 Algorithme

L'implémentation suit exactement les 4 étapes de la CP marginale d'Angelopoulos & Bates (2021) :

**Étape 1 — Score de non-conformité.** Pour chaque observation $i$ de $D_{\text{calibration}}$ :

$$S_i = 1 - \hat{f}(x_i)_{y_i}$$

où $\hat{f}(x_i)_{y_i}$ est la probabilité softmax attribuée par la logistique à la vraie classe $y_i$.

**Étape 2 — Quantile corrigé.**

$$\hat{q} = \text{Quantile}\left(\{S_1, \ldots, S_n\},\ \frac{\lceil(n+1)(1-\alpha)\rceil}{n}\right)$$

avec $\alpha = 0.05$ et $n = 2\,500$.

**Étape 3 — Ensemble de prédiction.** Pour chaque observation de $D_{\text{test}}$ :

$$C(x) = \{y : \hat{f}(x)_y \geq 1 - \hat{q}\}$$

**Étape 4 — Évaluation.** Calculer le pourcentage d'observations test dont la vraie classe est dans l'ensemble.

### 3.2 Paramètres obtenus

| Paramètre | Valeur |
|---|---|
| $\alpha$ | 0.05 |
| $n_{\text{calibration}}$ | 2 500 |
| Niveau du quantile | 0.950400 |
| $\hat{q}$ | 0.8552 |
| Seuil de score ($1 - \hat{q}$) | 0.1449 |

---

## 4. Résultats

### 4.1 Métriques globales

| Métrique | Valeur | Attendu |
|---|---|---|
| **Couverture marginale** | **95,00 %** | 95 % (± 2 %) |
| Couverture macro | 95,00 % | ≈ 95 % (classes équilibrées) |
| Taille moyenne des ensembles | 1,59 | — |

La couverture marginale est **exactement** à la cible de 95 %. La couverture macro est identique car les classes sont parfaitement équilibrées (250 observations par classe dans le test).

### 4.2 Couverture par classe

| Classe | $n_{\text{test}}$ | Couverture |
|---|---|---|
| 0 | 250 | 96,80 % ✅ |
| 1 | 250 | 94,40 % ✅ |
| 2 | 250 | 93,60 % ✅ |
| 3 | 250 | 95,20 % ✅ |
| 4 | 250 | 97,60 % ✅ |
| 5 | 250 | 94,80 % ✅ |
| 6 | 250 | 94,80 % ✅ |
| 7 | 250 | 93,20 % ✅ |
| 8 | 250 | 94,80 % ✅ |
| 9 | 250 | 94,80 % ✅ |

Toutes les classes sont dans l'intervalle [93 %, 98 %]. Les légères variations sont dues à la variabilité d'échantillonnage et à la géométrie des classes (les classes dont les centres sont proches ont un chevauchement plus important).

### 4.3 Distribution des tailles des ensembles

| Taille | Observations | Proportion |
|---|---|---|
| 1 | 1 036 | 41,4 % |
| 2 | 1 440 | 57,6 % |
| 3 | 24 | 1,0 % |

La grande majorité des prédictions contiennent 1 ou 2 espèces, ce qui montre que les ensembles restent informatifs.

---

## 5. Interprétation

### 5.1 Validation de l'implémentation

La couverture marginale de 95,00 % confirme que notre implémentation des 4 étapes de la CP (score de non-conformité, quantile corrigé, construction des ensembles, évaluation) est **correcte**. Le code peut être appliqué en confiance aux données réelles de Pl@ntNet.

### 5.2 Couverture marginale = couverture macro sur données équilibrées

Sur ce dataset synthétique, les deux métriques coïncident (95,00 % chacune). C'est attendu : quand chaque classe a le même nombre d'observations, la moyenne sur les observations (marginale) est identique à la moyenne sur les classes (macro). La différence entre ces deux métriques n'apparaît que sur des données déséquilibrées — ce qui est le cas des données Pl@ntNet avec la longue traîne.

### 5.3 Ce que le toy example ne teste PAS

Ce sanity check valide la logique algorithmique, mais il ne teste pas les problèmes spécifiques aux données réelles : la troncature des scores softmax (observations manquantes), le déséquilibre extrême entre espèces (longue traîne), et la question de l'échangeabilité (données crowd vs expert).

---

## 6. Figures produites

| Figure | Description |
|---|---|
| `fig_sanity_check.png` | Vue d'ensemble : nuage de points 2D des 10 gaussiennes, couverture par classe, distribution des tailles. |
| `fig_scores_toy.png` | Distribution des scores de non-conformité sur $D_{\text{calibration}}$ avec le quantile $\hat{q}$. |

---

## 7. Conclusion

L'implémentation de la CP marginale est **validée** sur un problème synthétique contrôlé. La couverture de 95,00 % correspond exactement à la cible théorique. Ce résultat nous autorise à appliquer le même code aux données Pl@ntNet en toute confiance, sachant que les écarts observés sur les données réelles sont attribuables aux spécificités du dataset (troncature, longue traîne) et non à un bug d'implémentation.

---

## 8. Références

- Angelopoulos, A. N. & Bates, S. (2021). *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*. arXiv:2107.07511.
