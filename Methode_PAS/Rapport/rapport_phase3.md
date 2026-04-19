# Rapport Phase 3 : Méthode PAS (Partitioned Adaptive Sets)

## Projet Pl@ntNet-CP — Prédiction Conformelle et Longue Traîne

**Auteur** : Dossou - Branche `dev4`  
**Date** : Avril 2026  
**Encadrants** : Joseph Salmon, Christophe Botella, Jean-Baptiste Fermanian

---

## 1. Objectif de la Phase 3

La Phase 2 (Conformal Prediction marginale) a mis en évidence un problème fondamental : la garantie de couverture à 95 % est une garantie **marginale**, c'est-à-dire en moyenne sur toutes les espèces. Or, le classifieur Pl@ntNet est biaisé en faveur des espèces communes, ce qui signifie que les espèces rares qui constituent la majorité de la biodiversité peuvent ne pas atteindre cette couverture cible.

L'objectif de la Phase 3 est d'implémenter la méthode **PAS (Partitioned Adaptive Sets)**, décrite dans Ding et al. (2025), pour corriger cette iniquité en calibrant séparément les espèces rares et communes.

---

## 2. Méthodologie

### 2.1 Principe de PAS

Au lieu de calculer un unique quantile $\hat{q}$ sur l'ensemble des données de calibration, PAS procède en trois étapes :

1. **Partitionnement** : les espèces sont classées en groupes selon leur prévalence dans le jeu de calibration (nombre d'observations).
2. **Calibration par groupe** : un quantile $\hat{q}_g$ est calculé indépendamment pour chaque groupe $g$.
3. **Prédiction adaptative** : pour une nouvelle image, chaque espèce candidate est évaluée avec le seuil de **son propre groupe**.

### 2.2 Définition des groupes

Nous avons défini deux groupes basés sur un seuil de rareté $n_{\text{seuil}} = 10$ :

- **Groupe "Rare"** : espèces avec moins de 10 observations en calibration.
- **Groupe "Commun"** : espèces avec 10 observations ou plus.

### 2.3 Correction des observations manquantes

Un point critique identifié lors de l'implémentation : certaines observations ont leur vraie espèce **absente** du fichier `ai_scores_all.csv` (score softmax inférieur au seuil de troncature de 0.001). Ces observations étaient silencieusement ignorées dans le calcul de non-conformité, ce qui biaisait le quantile vers des valeurs trop basses.

**Correction appliquée** : les observations dont la vraie espèce est absente des scores reçoivent un score de non-conformité $s_i = 1.0$ (échec total du modèle). Cela pousse le quantile vers une valeur plus conservative et permet d'atteindre la couverture cible.

---

## 3. Statistiques descriptives

### 3.1 Données utilisées

| Statistique | Valeur |
|---|---|
| Observations totales (test) | 10 812 |
| Espèces uniques dans le test | 2 401 |
| Espèces Rares (< 10 obs) | 2 138 (89,1 %) |
| Espèces Communes (≥ 10 obs) | 263 (10,9 %) |
| Observations de test Rares | 6 281 (58,1 %) |
| Observations de test Communes | 4 531 (41,9 %) |

Ce tableau confirme la structure de **longue traîne** : 89,1 % des espèces sont rares, mais elles ne représentent que 58,1 % des observations (les espèces communes ont davantage d'images par espèce).

### 3.2 Quantiles adaptatifs

| Paramètre | Groupe Rare | Groupe Commun |
|---|---|---|
| $\hat{q}_g$ | 0,9681 | 0,9927 |
| Seuil de score ($1 - \hat{q}_g$) | 0,0319 | 0,0073 |

Le seuil de score est plus bas (plus permissif) pour les deux groupes par rapport à la Baseline, ce qui reflète l'inclusion des observations manquantes dans la calibration.

### 3.3 Couverture : Baseline vs PAS

| Métrique | Baseline | PAS | Δ |
|---|---|---|---|
| **Couverture globale** | 91,53 % | **95,17 %** | +3,64 pp |
| Couverture Rares | 91,11 % | **94,68 %** | +3,57 pp |
| Couverture Communes | 92,20 % | **95,85 %** | +3,65 pp |
| Taille moyenne | 1,28 | 2,40 | +1,12 |

**Résultat principal** : PAS atteint la couverture cible de 95 % au niveau global (95,17 %). La couverture des espèces rares passe de 91,11 % à 94,68 %, une amélioration de +3,57 points de pourcentage.

### 3.4 Distribution de la taille des ensembles de prédiction

| Taille | Nombre | Proportion |
|---|---|---|
| 1 (prédiction unique) | 3 928 | 36,3 % |
| 2 | 3 123 | 28,9 % |
| 3 | 1 732 | 16,0 % |
| 4 | 941 | 8,7 % |
| 5 | 485 | 4,5 % |
| 6 | 250 | 2,3 % |
| 7 | 135 | 1,2 % |
| ≥ 8 | 218 | 2,0 % |

65,2 % des prédictions contiennent au plus 2 espèces, et 36,3 % donnent une prédiction unique, ce qui montre que les ensembles restent informatifs malgré l'élargissement.

### 3.5 Détail par groupe

| Statistique | Rares | Communes |
|---|---|---|
| Taille moyenne | 2,42 | 2,38 |
| Taille médiane | 2 | 2 |
| Taille maximale | 14 | 21 |
| % taille = 1 | 37,6 % | 34,6 % |
| % taille ≤ 2 | 64,8 % | 65,7 % |
| % taille ≥ 5 | 10,8 % | 9,0 % |
| Observations non couvertes | 334 | 188 |

---

## 4. Figures produites

1. **fig1_distribution_non_conformite.png** — Distribution des scores de non-conformité par groupe, avec les quantiles adaptatifs. On observe le pic à $s_i = 1.0$ correspondant aux observations manquantes.

2. **fig2_couverture_baseline_vs_pas.png** — Comparaison de la couverture entre Baseline et PAS, globalement et par groupe. La ligne verte pointillée représente la cible de 95 %.

3. **fig3_taille_ensembles_pas.png** — Distribution de la taille des ensembles de prédiction pour chaque groupe.

---

## 5. Discussion

### 5.1 Compromis couverture–efficacité

PAS atteint la couverture cible au prix d'ensembles plus larges (2,40 vs 1,28). Ce compromis est inhérent à la prédiction conformelle : une couverture plus élevée nécessite des ensembles plus prudents. Néanmoins, 36,3 % des prédictions restent de taille 1, ce qui signifie que le modèle reste très précis sur plus d'un tiers des images.

### 5.2 Couverture des rares encore légèrement sous-cible

La couverture des espèces rares (94,68 %) est très proche de 95 % mais ne l'atteint pas exactement. Cela peut s'expliquer par le fait qu'avec seulement 2 groupes, la partition n'est pas assez fine pour capturer toute l'hétérogénéité des espèces rares. Une extension avec 3 ou 4 groupes de rareté pourrait améliorer ce point.

### 5.3 Impact de la correction des observations manquantes

Cette correction est le facteur dominant dans l'amélioration de la couverture. Sans elle, les deux méthodes (Baseline et PAS) échouent à atteindre 95 %. C'est un résultat important : la qualité de la calibration dépend de l'inclusion de **tous** les cas, y compris les échecs du modèle.

---

## 6. Fichiers produits

| Fichier | Description |
|---|---|
| `pas_method.py` | Script principal de la Phase 3 |
| `resultats_pas.csv` | Ensembles de prédiction pour chaque observation test |
| `comparaison_baseline_pas.csv` | Tableau comparatif Baseline vs PAS |
| `figures/fig1_distribution_non_conformite.png` | Distribution des scores par groupe |
| `figures/fig2_couverture_baseline_vs_pas.png` | Comparaison de couverture |
| `figures/fig3_taille_ensembles_pas.png` | Distribution des tailles |

---

## 7. Références

- Ding, T., Fermanian, J.-B., Salmon, J. (2025). *Conformal Prediction for Long-Tailed Classification*. arXiv:2507.06867.
- Angelopoulos, A. N., & Bates, S. (2021). *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*. arXiv:2107.07511.
