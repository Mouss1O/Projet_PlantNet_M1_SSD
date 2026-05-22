# Rapport Phase 2 : Split Conformal Prediction

**Auteur :** Firda  
**Projet :** Pl@ntNet - Prédiction conformelle pour espèces rares  
**Date :** Avril 2025  

---

## 1. Introduction

Pl@ntNet est une application mobile de reconnaissance de plantes qui permet d'identifier plus de 75 000 espèces. Cependant, pour les espèces rares, le modèle manque de données d'entraînement, ce qui rend ses prédictions incertaines.

La prédiction conformelle permet de transformer les scores d'un classifieur en ensembles de prédiction avec une garantie théorique sur le taux d'erreur.

## 2. Objectifs

- Calculer des scores de non-conformité
- Déterminer un seuil garantissant une couverture de 95%
- Construire des ensembles de prédiction
- Évaluer la couverture observée

## 3. Méthodologie

### 3.1 Données

| Ensemble | Observations |
|----------|--------------|
| Calibration | 10 812 |
| Test | 10 812 |

### 3.2 Algorithme

1. **Score de non-conformité :** $S_i = 1 - \hat{p}_{y_i}$
2. **Seuil :** $\hat{q} = \text{quantile}(S_1, \ldots, S_n, \frac{(n+1)(1-\alpha)}{n})$
3. **Ensemble :** $C(X) = \{ y : \hat{p}_y \geq 1 - \hat{q} \}$

## 4. Résultats

### 4.1 Paramètres

| Paramètre | Valeur |
|-----------|--------|
| $\alpha$ | 0.05 |
| $n_{\text{calibration}}$ | 10 402 |
| $\hat{q}$ | 0.847 |
| Seuil sur scores | 0.153 |

### 4.2 Métriques

| Métrique | Valeur | Objectif |
|----------|--------|----------|
| Couverture | 91.5% | 95% |
| Taille moyenne | 1.28 | - |
| Ensembles vides | 0% | 0% |

### 4.3 Distribution

| Taille | Observations | Pourcentage |
|--------|--------------|-------------|
| 1 | 8 163 | 75.5% |
| 2 | 2 332 | 21.6% |
| 3 | 292 | 2.7% |
| 4 | 21 | 0.2% |
| 5 | 4 | 0.0% |

## 5. Discussion

La couverture observée (91.5%) est inférieure à l'objectif de 95%. Cette sous-couverture s'explique par :

- Présence d'espèces rares avec des scores faibles
- Seuil unique inadapté
- Non-échangeabilité des données

## 6. Conclusion

La méthode standard ne permet pas d'atteindre la couverture garantie de 95%.

## 7. Perspectives (Phase 3)

Implémentation de la méthode **PAS** (Ding et al., 2025) pour adapter le seuil aux espèces rares.

## 8. Références

- Angelopoulos & Bates (2021)
- Lefort et al. (2024)
- Ding et al. (2025)
