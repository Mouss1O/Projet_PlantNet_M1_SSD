# Rapport d'avancement
## Prédiction conformelle et base de données Pl@ntNet-CrowdSWE

**Groupe :** Dossou Agossou, Firdaousse Karimou, Moussa Diagne  

---

## Résumé

Ce document présente l'avancement de notre projet sur l'application de la prédiction conformelle aux données Pl@ntNet-CrowdSWE-v2. Nous avons structuré le travail en trois phases, toutes opérationnelles. Les résultats montrent que la méthode marginale standard n'atteint pas la couverture cible de 95 %, et que la méthode PAS corrige ce problème en calibrant séparément les espèces rares et communes.

---

## Phase 1 : Exploration et partitionnement des données

**Responsable :**

Nous avons travaillé sur le dataset Pl@ntNet-CrowdSWE-v2 (5,56 millions d'observations, 57 660 espèces). À partir du fichier `ground_truth.json`, nous avons isolé les **21 624 observations validées par des experts** (labels ≠ −1), couvrant 3 082 espèces.

Le partitionnement a été réalisé avec un **split 50/50** :

| Ensemble | Observations |
|---|---|
| Calibration | 10 812 |
| Test | 10 812 |

Vérifications effectuées : aucun doublon, aucune fuite entre calibration et test, identifiants uniques.

---

## Phase 2 : Conformal Prediction marginale (Baseline)

**Responsable :**

Nous avons implémenté la méthode standard d'Angelopoulos & Bates (2021) avec α = 0.05 :

1. Score de non-conformité : $s_i = 1 - \hat{f}(x_i)_{y_i}$
2. Quantile corrigé : $\hat{q} = Q_{\lceil(n+1)(1-\alpha)\rceil / n}$
3. Ensemble de prédiction : $C(x) = \{y : \hat{f}(x)_y \geq 1 - \hat{q}\}$

| Métrique | Valeur |
|---|---|
| $\hat{q}$ | 0,847 |
| Couverture globale | **91,53 %** |
| Taille moyenne | 1,28 |

**Constat :** la couverture est inférieure à la cible de 95 %. Nous avons identifié la cause : les observations dont la vraie espèce a un score softmax < 0,001 (absente du fichier `ai_scores_all`) sont exclues du calcul de non-conformité, ce qui biaise le quantile vers une valeur trop basse.

---

## Phase 3 : Méthode PAS (Partitioned Adaptive Sets)

**Responsable :** Dossou Agossou

Nous avons implémenté la méthode PAS

### Calibration par groupe de rareté

Les espèces sont partitionnées en deux groupes (seuil : 10 observations) :

| Groupe | Espèces | Observations test | $\hat{q}_g$ | Seuil de score |
|---|---|---|---|---|
| Rare (< 10 obs) | 2 138 (89 %) | 6 281 (58 %) | 0,9681 | 0,0319 |
| Commun (≥ 10 obs) | 263 (11 %) | 4 531 (42 %) | 0,9927 | 0,0073 |

### Résultats comparatifs

| Métrique | Baseline | PAS | Δ |
|---|---|---|---|
| **Couverture globale** | 91,53 % | **95,17 %** | +3,64 pp |
| Couverture Rares | 91,11 % | **94,68 %** | +3,57 pp |
| Couverture Communes | 92,20 % | **95,85 %** | +3,65 pp |
| Taille moyenne | 1,28 | 2,40 | +1,12 |

PAS atteint la couverture cible au prix d'ensembles plus larges. 36,3 % des prédictions restent de taille 1.
