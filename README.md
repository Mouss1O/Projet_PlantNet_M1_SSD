**Auteur :** Firda\
**Date :** Avril 2025\
**Projet :** [Pl\@ntNet](mailto:Pl@ntNet){.email} - Prédiction conformelle pour espèces rares

------------------------------------------------------------------------

## Objectif

Implémenter la méthode de split conformal prediction (Angelopoulos & Bates, 2021) pour obtenir des ensembles de prédiction avec une garantie de couverture à 95%.

------------------------------------------------------------------------

## Méthode

| Étape | Description |
|----|----|
| 1 | Calcul des scores de non-conformité : S = 1 - score_vraie_espece |
| 2 | Calcul du seuil q_hat = quantile(S, (n+1)(1-α)/n) |
| 3 | Construction des ensembles : C(X) = { espèces avec score ≥ 1 - q_hat } |
| 4 | Évaluation de la couverture et de la taille |

------------------------------------------------------------------------

## Résultats

### Paramètres

-   α = 0.05 (couverture cible = 95%)
-   n_calibration = 10 402
-   q_hat = 0.847
-   Seuil sur les scores = 0.153

### Métriques

| Métrique        | Valeur | Objectif |
|-----------------|--------|----------|
| Couverture      | 91.5%  | 95%      |
| Taille moyenne  | 1.28   | \-       |
| Taille médiane  | 1      | \-       |
| Ensembles vides | 0%     | 0%       |

### Distribution des tailles

| Taille    | Observations | Pourcentage |
|-----------|--------------|-------------|
| 1 espèce  | 8 163        | 75.5%       |
| 2 espèces | 2 332        | 21.6%       |
| 3 espèces | 292          | 2.7%        |
| 4 espèces | 21           | 0.2%        |
| 5 espèces | 4            | 0.0%        |

------------------------------------------------------------------------

## Interprétation

**Couverture insuffisante (91.5% \< 95%)**

La méthode standard échoue à atteindre la couverture garantie à cause des **espèces rares** : - Elles ont des scores plus faibles - Le seuil unique ne leur est pas adapté

**Taille des ensembles :** très petite (1.28 en moyenne) car le modèle est très confiant mais parfois trop.

------------------------------------------------------------------------

## Conclusion

La méthode standard ne fonctionne pas parfaitement sur ce jeu de données.\
**Phase 3 :** Implémentation de la méthode PAS (Ding et al., 2025) pour adapter le seuil aux espèces rares.

------------------------------------------------------------------------

## Structure du dépôt

```         
Projet_PlantNet_M1_SSD/
│
├── README.md
│
├── scripts/
│   ├── non_conformity.py
│   ├── quantile.py
│   ├── ensembles_prediction.py
│   ├── evaluations_finale.py
│   └── verification.py
│
├── results/
│   ├── non_conformity.csv
│   ├── seuil.csv
│   ├── resultats.csv
│   └── metrics_phase2.csv
│
├── rapport/
│
├── data/
│   └── split_random/
│       ├── calibration.csv
│       └── test.csv
│
├── PlantNet_Convertis/
├── PlantNet_classification/
├── fichiers_utilises/
└── Travail_Firda/
```
