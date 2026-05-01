# Rapport Phase 3 : Correction du Biais et Test de la CP Conditionnelle

## Projet Pl@ntNet-CP — Prédiction Conformelle et Longue Traîne

**Auteur** : Dossou Agossou 
**Date** : Avril 2026  
**Encadrants** : Joseph Salmon, Christophe Botella, Jean-Baptiste Fermanian

---

## 1. Objectifs

Cette phase poursuit deux objectifs :

1. **Corriger le biais** dans le calcul des scores de non-conformité afin d'atteindre la couverture marginale cible de 95 %.
2. **Tester la CP conditionnelle** (un quantile par espèce) et démontrer son inefficacité pratique due à la taille excessive des ensembles de prédiction.

---

## 2. Volet 1 : Correction du Biais

### 2.1 Identification du problème

Le fichier `ai_scores_all.csv` ne conserve que les espèces ayant un score softmax ≥ 0.001 pour chaque observation. Les espèces avec un score inférieur sont **tronquées** du fichier.

Conséquence : pour certaines observations, la **vraie espèce** (celle validée par l'expert) est absente du fichier de scores. Dans le calcul initial de non-conformité, ces observations sont silencieusement ignorées, ce qui introduit un **biais de sélection** : le quantile est estimé uniquement sur les cas "faciles" (ceux où le modèle a au moins détecté la bonne espèce).

### 2.2 Diagnostic quantitatif

| Statistique | Valeur |
|---|---|
| Observations en calibration | 10 812 |
| Vraie espèce trouvée dans les scores | 10 402 (96,2 %) |
| Vraie espèce **absente** (score < 0.001) | 410 (3,8 %) |
| Espèces uniques concernées | 149 |
| % d'espèces rares parmi les manquantes | 46,1 % |
| Prévalence médiane des espèces concernées | 13 obs/espèce |

Ces 410 observations ne sont pas des erreurs de données : ce sont les cas où le modèle Pl@ntNet a **complètement échoué** à reconnaître la bonne espèce. Elles concernent disproportionnellement des espèces rares, confirmant le biais du modèle en faveur des espèces communes.

### 2.3 Correction appliquée

Le score de non-conformité est défini par $S_i = 1 - \hat{f}(x_i)_{y_i}$, où $\hat{f}(x_i)_{y_i}$ est le score softmax attribué à la vraie espèce $y_i$.

Pour les observations dont la vraie espèce est absente du fichier de scores :

- Le score softmax est inférieur à 0.001 (seuil de troncature du dataset).
- Donc $S_i = 1 - (\text{valeur} < 0.001) \approx 1.0$.
- On assigne $S_i = 1.0$, ce qui correspond à un **échec total du modèle**.

Ce choix est **conservatif** : la valeur exacte est comprise entre 0.999 et 1.0, mais la différence est négligeable en pratique.

### 2.4 Impact de la correction

| Métrique | Sans correction | Avec correction | Δ |
|---|---|---|---|
| $n_{\text{calibration}}$ | 10 402 | 10 812 | +410 |
| $\hat{q}$ | 0.8476 | 0.9789 | +0.1313 |
| Seuil de score | 0.1524 | 0.0211 | −0.1313 |
| **Couverture marginale** | **91,54 %** | **95,26 %** | **+3,72 pp** |
| Taille moyenne | 1.28 | 2.66 | +1.39 |

L'inclusion des 410 observations avec $S_i = 1.0$ pousse le quantile de 0.8476 à 0.9789. Le seuil de score baisse de 0.15 à 0.02, ce qui permet d'accepter davantage d'espèces candidates dans les ensembles de prédiction. La couverture remonte de 91,54 % à 95,26 %, atteignant la cible théorique.

Le prix à payer est un élargissement des ensembles de prédiction (de 1.28 à 2.66 espèces en moyenne), ce qui reflète le compromis fondamental couverture–efficacité de la prédiction conformelle.

---

## 3. Volet 2 : Test de la CP Conditionnelle

### 3.1 Principe

La CP conditionnelle vise à garantir la couverture **pour chaque espèce individuellement** en calculant un quantile $\hat{q}_k$ par espèce $k$. Pour une nouvelle image, chaque espèce candidate est évaluée avec le seuil de **sa propre espèce** :

$$C_{\text{cond}}(x) = \{y : \hat{f}(x)_y \geq 1 - \hat{q}_y\}$$

### 3.2 Problème fondamental

La médiane du nombre d'observations par espèce en calibration est de **2**. Cela signifie que pour la majorité des espèces, le quantile est estimé sur 2 ou 3 points.

| Statistique | Valeur |
|---|---|
| Espèces avec quantile calculé | 2 458 |
| Médiane des $n_k$ | 2 |
| Espèces avec $\hat{q}_k = 1.0$ | 149 (6,1 %) |
| Moyenne des $\hat{q}_k$ | 0.4686 |

Un quantile à 95 % estimé sur 2 observations n'a aucune fiabilité statistique. Pour certaines espèces, le seuil est trop bas (sous-couverture) ; pour d'autres, il est à 1.0 (tous les candidats acceptés, ensembles gigantesques).

### 3.3 Comparaison des 3 méthodes

| Métrique | Standard | PAS | Conditionnelle |
|---|---|---|---|
| **Couv. marginale** | 95,26 % | 95,18 % | 83,56 % |
| **Couv. macro** | 94,79 % | 94,37 % | 77,74 % |
| Couv. marg. Rares | 95,21 % | 94,70 % | 77,46 % |
| Couv. marg. Communes | 95,32 % | 95,85 % | 92,01 % |
| Couv. macro Rares | 94,61 % | 94,07 % | 76,07 % |
| Couv. macro Communes | 96,22 % | 96,84 % | 91,32 % |
| **Taille moyenne** | **2,66** | **2,42** | **11,83** |
| Taille médiane | 2 | 2 | 7 |
| Taille Rares | 2,74 | 2,43 | 12,93 |
| Taille Communes | 2,56 | 2,40 | 10,32 |

### 3.4 Analyse des résultats

**Couverture marginale** : Standard et PAS atteignent toutes deux ~95 %, conformément à la garantie théorique. La Conditionnelle échoue avec seulement 83,56 % — l'instabilité des quantiles par espèce produit à la fois de la sous-couverture et des ensembles inutilement larges.

**Couverture macro** : Standard (94,79 %) et PAS (94,37 %) sont très proches. La correction du biais est le facteur dominant — une fois le biais corrigé, les deux méthodes atteignent des performances comparables. La Conditionnelle s'effondre à 77,74 %.

**Taille des ensembles** : c'est le résultat le plus frappant. La Conditionnelle produit des ensembles **4,4 fois plus grands** que le Standard (11,83 vs 2,66 espèces en moyenne, médiane à 7). Pour un utilisateur de Pl@ntNet, recevoir une liste de 12 espèces possibles au lieu de 2 rend la prédiction inutile.

**PAS comme compromis** : PAS produit des ensembles légèrement plus petits que le Standard (2,42 vs 2,66) tout en maintenant la couverture marginale. Sur les espèces communes, PAS améliore la couverture macro (96,84 % vs 96,22 %).

---

## 4. Distinction fondamentale : méthodes vs métriques

Un point conceptuel important clarifié au cours de cette phase : les **méthodes** (Standard, PAS, Conditionnelle) et les **métriques** (couverture marginale, macro, conditionnelle) sont deux axes indépendants.

| | Méthode Standard | Méthode PAS | Méthode Conditionnelle |
|---|---|---|---|
| **Nombre de quantiles** | 1 (global) | 2 (par groupe) | 2 458 (par espèce) |
| **Objectif** | Couv. marginale ≥ 95 % | Couv. macro ↑ | Couv. conditionnelle |
| **Garantie théorique** | Marginale ≥ 1−α | Marginale ≈ 1−α | Conditionnelle (en théorie) |

N'importe quelle méthode peut être évaluée avec n'importe quelle métrique. Dans notre cas :
- Standard et PAS atteignent la couverture marginale cible.
- La Conditionnelle échoue sur toutes les métriques à cause de l'insuffisance de données par espèce.

---

## 5. Figures produites

| Figure | Description |
|---|---|
| `fig_correction_biais_distribution.png` | Distribution des scores de non-conformité avant et après correction. On observe le pic à $S_i = 1.0$ et le déplacement du quantile. |
| `fig_correction_biais_comparaison.png` | Barplot comparant couverture et taille des ensembles avant/après correction. |
| `fig_prevalence_manquantes.png` | Histogramme de la prévalence des espèces concernées par le biais, confirmant la surreprésentation des espèces rares. |
| `fig_comparaison_3_methodes.png` | Vue d'ensemble des 3 méthodes : couvertures marginale/macro, distribution des tailles, boxplot. |
| `fig_couverture_par_groupe_3_methodes.png` | Couverture marginale et macro par groupe (Rares/Communes) pour les 3 méthodes. |

---

## 6. Conclusion

### 6.1 Le correctif est le facteur dominant

L'inclusion des observations manquantes (avec $S_i = 1.0$) est la correction la plus impactante de tout le projet. Sans elle, aucune méthode (Standard, PAS, Conditionnelle) n'atteint la couverture cible. Avec elle, Standard et PAS atteignent toutes deux ~95 %.

Ce résultat souligne l'importance de la qualité du pipeline de données en amont de toute méthode statistique.

### 6.2 PAS apporte un gain modéré

Dans notre configuration (2 groupes, seuil à 10), PAS et Standard ont des performances très similaires une fois le biais corrigé. PAS améliore légèrement la couverture macro des espèces communes (96,84 % vs 96,22 %) et produit des ensembles légèrement plus petits (2,42 vs 2,66).

### 6.3 La CP Conditionnelle est contre-productive

La CP Conditionnelle est théoriquement la plus ambitieuse mais pratiquement la pire : couverture marginale de 83,56 %, couverture macro de 77,74 %, et des ensembles 4,4 fois plus grands. Avec une médiane de 2 observations par espèce en calibration, les quantiles individuels sont statistiquement instables.

### 6.4 Le défi structurel des espèces très rares

Le scatter plot couverture vs prévalence révèle le vrai défi : les espèces avec 1 à 3 observations ont une couverture variant de 0 % à 100 %, quelle que soit la méthode. C'est une limite fondamentale : aucune méthode statistique ne peut fournir une garantie fiable quand les données sont aussi rares.

---

## 7. Fichiers produits

| Fichier | Description |
|---|---|
| `correction_biais.py` | Script de correction du biais (avant/après) |
| `cp_conditionnelle.py` | Script de comparaison des 3 méthodes |
| `correction_biais_comparaison.csv` | Tableau comparatif avant/après correction |
| `comparaison_3_methodes.csv` | Tableau comparatif des 3 méthodes |

---

## 8. Références

- Angelopoulos, A. N. & Bates, S. (2021). *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*. arXiv:2107.07511.
- Ding, T., Fermanian, J.-B. & Salmon, J. (2025). *Conformal Prediction for Long-Tailed Classification*. arXiv:2507.06867.
- Lefort, T. et al. (2024). *Pl@ntNet collaborative learning: South-Western-Europe dataset*. arXiv:2406.03356.
