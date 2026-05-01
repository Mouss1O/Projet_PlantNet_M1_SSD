# Rapport Phase 2 : Couverture Marginale vs Couverture Macro

## Projet Pl@ntNet-CP — Prédiction Conformelle et Longue Traîne

**Auteur** : Firdaousse Karimou 
**Date** : Avril 2026  
**Encadrants** : Joseph Salmon, Christophe Botella, Jean-Baptiste Fermanian

---

## 1. Objectif

Répondre à l'exigence de M. Salmon sur la clarification des mesures de performance : distinguer la **couverture marginale** (moyenne sur les observations) de la **couverture macro** (moyenne sur les espèces), et comparer ces deux métriques entre la CP Standard et la méthode PAS.

Le prof a précisé (mail du 21/04/2026) : *"Pour les couvertures marginales, les deux méthodes PAS et standard\_CP devrait atteindre à peu près la même couverture marginale par construction. Par contre PAS devrait avoir une meilleure couverture macro."*

---

## 2. Rappel : Marginale vs Macro

### 2.1 Couverture marginale

$$\widehat{\text{Couv}}_{\text{marginale}} = \frac{1}{n_{\text{test}}} \sum_{i=1}^{n_{\text{test}}} \mathbb{1}\{y_i \in C(x_i)\}$$

C'est le taux de succès **global**, en comptant chaque observation avec le même poids. Les espèces communes, qui ont beaucoup d'observations, dominent cette métrique. Une espèce rare avec 3 observations pèse 3 fois moins qu'une espèce commune avec 50 observations.

### 2.2 Couverture macro

$$\widehat{\text{Couv}}_{\text{macro}} = \frac{1}{K} \sum_{k=1}^{K} \widehat{\text{Couv}}_k \quad \text{où} \quad \widehat{\text{Couv}}_k = \frac{1}{n_k} \sum_{i : y_i = k} \mathbb{1}\{y_i \in C(x_i)\}$$

On calcule d'abord la couverture **de chaque espèce** individuellement, puis on fait la moyenne. Chaque espèce compte autant, quelle que soit sa prévalence. Cette métrique révèle les faiblesses sur les espèces rares.

### 2.3 Quand les deux métriques divergent-elles ?

Sur des données **équilibrées** (même nombre d'observations par classe), les deux métriques sont identiques — c'est ce qu'on a observé dans le sanity check (Phase 1 : marginale = macro = 95,00 %).

Sur des données **déséquilibrées** (longue traîne de Pl@ntNet), les deux métriques peuvent diverger significativement. Une espèce rare mal couverte fait chuter la couverture macro sans affecter la couverture marginale.

---

## 3. Données et méthodes

### 3.1 Données

Les deux méthodes (Standard et PAS) sont appliquées sur les mêmes données, **avec la correction du biais** (observations manquantes assignées à $S_i = 1.0$) :

| Ensemble | Taille |
|---|---|
| Calibration | 10 812 observations |
| Test | 10 812 observations |
| Espèces dans le test | 2 401 |
| Espèces Rares (< 10 obs) | 2 138 (89,0 %) |
| Espèces Communes (≥ 10 obs) | 263 (11,0 %) |

### 3.2 Quantiles calculés

| Méthode | Quantile(s) | Seuil(s) de score |
|---|---|---|
| Standard | $\hat{q} = 0.9789$ | $1 - \hat{q} = 0.0211$ |
| PAS Rare | $\hat{q}_{\text{rare}} = 0.9683$ | $1 - \hat{q} = 0.0317$ |
| PAS Commun | $\hat{q}_{\text{commun}} = 0.9931$ | $1 - \hat{q} = 0.0069$ |

---

## 4. Résultats

### 4.1 Comparaison globale

| Métrique | Standard | PAS | Δ |
|---|---|---|---|
| **Couverture marginale** | **95,26 %** | **95,18 %** | −0,08 pp |
| **Couverture macro** | **94,79 %** | **94,37 %** | −0,42 pp |
| Taille moyenne | 2,66 | 2,42 | −0,24 |

Les deux méthodes atteignent la couverture marginale cible (~95 %), conformément à la théorie. La couverture macro est également très proche entre les deux méthodes.

PAS produit des ensembles légèrement plus petits (2,42 vs 2,66), ce qui est un avantage en termes d'efficacité.

### 4.2 Comparaison par groupe

**Groupe Rare (2 138 espèces) :**

| Métrique | Standard | PAS |
|---|---|---|
| Couverture marginale | 95,21 % | 94,70 % |
| Couverture macro | 94,61 % | 94,07 % |

**Groupe Commun (263 espèces) :**

| Métrique | Standard | PAS |
|---|---|---|
| Couverture marginale | 95,32 % | 95,85 % |
| Couverture macro | 96,22 % | 96,84 % |

Sur les espèces **communes**, PAS améliore la couverture macro (96,84 % vs 96,22 %, soit +0,62 pp). Sur les espèces **rares**, Standard est légèrement meilleur (94,61 % vs 94,07 %).

### 4.3 Impact par espèce

| Impact de PAS | Nombre d'espèces | Proportion |
|---|---|---|
| Couverture améliorée | 21 | 0,9 % |
| Couverture dégradée | 29 | 1,2 % |
| Couverture inchangée | 2 351 | 97,9 % |

Pour la grande majorité des espèces (97,9 %), les deux méthodes produisent exactement la même couverture. L'impact de PAS est concentré sur un petit nombre d'espèces.

### 4.4 Distribution des couvertures par espèce

| Statistique | Standard | PAS |
|---|---|---|
| Moyenne (= couv. macro) | 94,79 % | 94,37 % |
| Médiane | 100,00 % | 100,00 % |
| Écart-type | 18,75 % | 19,51 % |
| Minimum | 0,00 % | 0,00 % |
| % espèces à 100 % | 88,5 % | 88,2 % |
| % espèces à 0 % | 2,8 % | 3,1 % |
| % espèces ≥ 95 % | 89,4 % | 88,7 % |
| % espèces < 80 % | 7,5 % | 8,1 % |

La distribution est fortement concentrée à 100 % (près de 89 % des espèces sont parfaitement couvertes) avec une queue de distribution vers les basses couvertures. Les espèces à 0 % de couverture (~3 %) sont celles pour lesquelles le modèle échoue systématiquement.

---

## 5. Analyse des figures

### 5.1 Boxplot des couvertures par espèce

Le boxplot montre que les deux méthodes ont des distributions quasi identiques, avec une médiane à 100 % et des outliers vers les basses couvertures. La couverture macro (losange rouge) est légèrement en dessous de la ligne des 95 % pour les deux méthodes.

Sur les espèces rares uniquement, la dispersion est similaire, avec quelques espèces à couverture nulle.

### 5.2 Scatter plot couverture vs prévalence

C'est la figure la plus informative. On observe :

- Les espèces avec **1 à 3 observations** ont une couverture très variable (entre 0 % et 100 %) pour les deux méthodes. C'est un effet mécanique : avec 1 observation, la couverture est soit 0 % soit 100 %.
- À partir de **10+ observations**, la couverture se stabilise au-dessus de 80 % pour la grande majorité des espèces.
- Les deux méthodes produisent des patterns très similaires, confirmant que la correction du biais est le facteur principal et non le choix de la méthode.

### 5.3 Barplot marginale vs macro

La figure confirme visuellement que les deux métriques sont très proches (~95 % marginale, ~94,5 % macro) pour les deux méthodes. L'écart marginale–macro (~0,5 pp) est faible dans notre cas car la correction du biais a déjà résolu l'essentiel du problème.

---

## 6. Discussion

### 6.1 Pourquoi Standard et PAS sont-ils si proches ?

Trois facteurs expliquent la similarité des résultats :

**La correction du biais domine.** L'ajout des 410 observations manquantes avec $S_i = 1.0$ pousse le quantile de 0.85 à 0.98 pour les deux méthodes. Cet effet est beaucoup plus important que la différence entre un quantile global et deux quantiles par groupe.

**Le seuil de rareté est peut-être trop grossier.** Avec seulement 2 groupes (< 10 vs ≥ 10), la partition ne capture pas toute l'hétérogénéité des espèces rares. Une espèce avec 1 observation et une espèce avec 9 observations sont traitées de la même manière.

**La prévalence dans le test est très faible.** La médiane est de 2 observations par espèce dans le test. Avec si peu de données, la couverture par espèce est intrinsèquement binaire (0 % ou 100 %), ce qui réduit l'espace pour que PAS puisse se différencier.

### 6.2 Couverture marginale ≈ couverture macro : un bon signe

L'écart faible entre marginale et macro (0,5 pp) indique que, une fois le biais corrigé, le modèle ne discrimine plus fortement entre espèces rares et communes. Le problème initial de sous-couverture n'était pas inhérent au modèle mais au pipeline de données.

### 6.3 Les espèces à couverture nulle

Environ 3 % des espèces (≈ 70 espèces) ont une couverture de 0 %, c'est-à-dire que le modèle échoue sur **toutes** leurs observations test. Ces espèces sont les plus rares et les plus difficiles à identifier. Aucune méthode de calibration ne peut résoudre ce problème — il faudrait améliorer le modèle de classification lui-même (plus de données d'entraînement, augmentation de données, few-shot learning).

---

## 7. Figures produites

| Figure | Description |
|---|---|
| `fig_boxplot_couverture_par_espece.png` | Distribution des couvertures par espèce (boxplot), toutes espèces et rares uniquement. |
| `fig_histogramme_couverture_par_espece.png` | Histogramme des couvertures par espèce, toutes espèces et rares. |
| `fig_couverture_vs_prevalence.png` | Scatter plot de la couverture en fonction du nombre d'observations par espèce. |
| `fig_marginale_vs_macro.png` | Barplot comparant couverture marginale et macro pour Standard et PAS. |
| `couverture_par_espece.csv` | Couverture individuelle de chaque espèce pour les deux méthodes. |

---

## 8. Conclusion

La couverture marginale et la couverture macro sont des métriques complémentaires qui répondent à des questions différentes. Sur les données Pl@ntNet-CrowdSWE :

- **Couverture marginale** : les deux méthodes atteignent ~95 %, conformément à la garantie théorique de la CP, une fois le biais des observations manquantes corrigé.
- **Couverture macro** : les deux méthodes sont proches (~94,5 %), ce qui indique que la correction du biais est le facteur dominant.
- **PAS** offre un avantage modeste sur les espèces communes (+0,62 pp de couverture macro) et produit des ensembles légèrement plus petits (2,42 vs 2,66).
- Le **vrai défi** reste les espèces avec très peu d'observations (1-3), pour lesquelles la couverture est intrinsèquement instable.

---

## 9. Références

- Angelopoulos, A. N. & Bates, S. (2021). *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*. arXiv:2107.07511.
- Ding, T., Fermanian, J.-B. & Salmon, J. (2025). *Conformal Prediction for Long-Tailed Classification*. arXiv:2507.06867.
