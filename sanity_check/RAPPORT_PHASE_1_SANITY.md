# Rapport Phase 1 : Sanity Check sur Dataset Synthétique

**Auteur :** Moussa
**Projet :** Pl@ntNet - Prédiction conformelle pour espèces rares
**Date :** Avril 2025

---

## 1. Introduction

Lors des phases précédentes du projet, l'application de la prédiction conformelle classique sur les données Pl@ntNet a abouti à une couverture observée de 91 %, en deçà de la cible théorique de 95 %. Le professeur a légitimement soulevé un doute : cet écart provient-il d'un bug dans notre implémentation, ou bien des spécificités du dataset Pl@ntNet (long tail, scores tronqués à 0,001, hétérogénéité des labels) ?

Pour trancher cette question, il est indispensable d'effectuer un **sanity check** : tester notre implémentation sur un dataset simple et contrôlé, où la théorie garantit une couverture exacte de 95 %. Si le code atteint la cible sur ce cas idéal, alors l'implémentation est validée et l'écart sur Pl@ntNet doit être imputé aux données.

## 2. Objectifs

- Valider mathématiquement notre implémentation de la prédiction conformelle
- Mesurer la stabilité de la méthode sur plusieurs réplications
- Confirmer ou infirmer l'hypothèse que le code est correct
- Isoler le code des données pour préparer la Phase 3 (correction du biais)

## 3. Méthodologie

### 3.1 Génération des données synthétiques

Nous utilisons `sklearn.datasets.make_classification` pour créer un dataset contrôlé de 10 000 observations réparties en 10 classes, avec 4 features informatives et 1 cluster par classe.

Le sujet initial mentionnait 2 variables explicatives, mais sklearn impose la contrainte $K \leq 2^{n_{informative}}$. Pour 10 classes, il faut donc au minimum 4 features informatives ($2^4 = 16 \geq 10$).

### 3.2 Démarche expérimentale

Les 10 000 observations sont divisées en deux blocs égaux. Le premier (Bloc A, 5 000 observations) sert à entraîner un modèle de régression logistique multinomiale. Le second (Bloc B, 5 000 observations) est lui-même découpé en deux sous-ensembles : un ensemble de calibration ($D_{cal}$, 2 500 observations) servant à calculer le quantile conforme, et un ensemble de test ($D_{test}$, 2 500 observations) pour évaluer la couverture obtenue.

### 3.3 Algorithme de prédiction conforme

Sur l'ensemble de calibration, on calcule pour chaque observation le score de non-conformité $s_i = 1 - \hat{p}(y_i \mid x_i)$, c'est-à-dire le complément à 1 de la probabilité prédite par le modèle pour la vraie classe. On en déduit le quantile conforme avec correction finie-sample :

$$\hat{q} = \text{quantile}\left(s_1, \ldots, s_n,\ \frac{\lceil(n+1)(1-\alpha)\rceil}{n}\right)$$

Sur l'ensemble de test, l'ensemble de prédiction associé à une observation $x$ est défini comme l'ensemble des classes dont la probabilité prédite dépasse le seuil $1 - \hat{q}$ :

$$\mathcal{C}(x) = \{ y : \hat{p}(y \mid x) \geq 1 - \hat{q} \}$$

La couverture observée est la fraction des observations test pour lesquelles la vraie classe $y$ appartient bien à $\mathcal{C}(x)$.

### 3.4 Réplications

L'expérience est répétée 10 fois avec des graines aléatoires distinctes (seeds 42 à 51) afin de mesurer la stabilité statistique de la méthode et d'éviter de tirer des conclusions hâtives à partir d'une seule exécution potentiellement chanceuse ou malchanceuse.

## 4. Résultats

### 4.1 Paramètres

Le niveau d'erreur retenu est $\alpha = 0{,}05$, correspondant à une couverture cible de 95 %. Les bornes acceptables sont fixées à [94 %, 96 %].

### 4.2 Résultats globaux

| Métrique | Valeur |
|----------|--------|
| Couverture moyenne | **94,87 %** |
| Écart-type | 0,43 % |
| Couverture min | 94,00 % |
| Couverture max | 95,56 % |
| Réplications dans [94 %, 96 %] | **10 / 10** |
| Implémentation | **VALIDÉE** |

Sur les 10 réplications, la couverture moyenne s'établit à 94,87 %, soit un écart de seulement 0,13 % par rapport à la cible théorique de 95 %. Toutes les réplications sans exception tombent dans la zone acceptable [94 %, 96 %], avec un écart-type très faible de 0,43 %.

La taille moyenne des ensembles de prédiction est de 4,29 classes (sur 10 possibles), avec un écart-type de 0,30. L'accuracy moyenne du modèle logistique est de 61,40 %.

## 5. Discussion

### 5.1 Validation de l'implémentation

La couverture moyenne de 94,87 % avec un écart-type de 0,43 % démontre que notre code est mathématiquement correct, atteignant la cible théorique avec une précision remarquable. Les 10 réplications sur 10 dans la zone acceptable confirment que la méthode est statistiquement stable, sans comportement erratique. Le quantile conforme avec correction finie-sample fonctionne donc comme prévu.

### 5.2 Comparaison avec les résultats Pl@ntNet

Sur le dataset synthétique, l'écart à la cible n'est que de 0,13 %, alors que sur Pl@ntNet il atteint 4 %. L'écart sur les données réelles est donc environ 30 fois supérieur à celui observé sur les données synthétiques. Cette différence quantitative confirme que le problème ne vient pas du code mais bien des spécificités du dataset Pl@ntNet.

### 5.3 Une démonstration intéressante de la valeur de la prédiction conformelle

Un résultat secondaire mérite d'être souligné : l'accuracy du modèle logistique n'est que de 61 %, ce qui signifie qu'il se trompe dans environ 4 cas sur 10. Pourtant, la prédiction conformelle parvient à garantir une couverture de 95 %. Comment ? Simplement en élargissant les ensembles de prédiction : la taille moyenne est de 4,29 classes sur 10 possibles. La prédiction conformelle compense ainsi la faiblesse du modèle en proposant plusieurs candidats, garantissant que la vraie classe soit incluse 95 % du temps.

Ce résultat illustre parfaitement la philosophie de la prédiction conformelle : elle ne corrige pas un mauvais modèle, elle quantifie honnêtement son incertitude.

### 5.4 Implications pour la suite du projet

Le sanity check ayant validé le code, l'écart de 4 % observé sur Pl@ntNet provient nécessairement d'une violation des hypothèses sur les données réelles. Le professeur a identifié la cause la plus probable dans son retour : les observations dont la vraie espèce a un score softmax inférieur à 0,001 sont absentes du fichier `ai_scores_all` et donc exclues du calcul de non-conformité, ce qui biaise le quantile vers une valeur trop basse.

Lors de la Phase 1 originelle de préparation des données, nous avions effectivement identifié que 3,8 % des observations expertes ont leur vraie classe absente du top-k stocké. Le correctif consistera à assigner $s_i = 1$ (score de non-conformité maximal) à ces observations plutôt que de les exclure du calcul.

## 6. Conclusion

Le sanity check sur dataset synthétique a permis de valider rigoureusement notre implémentation de la prédiction conformelle. Avec une couverture moyenne de 94,87 % sur 10 réplications, un écart-type de seulement 0,43 % et l'ensemble des réplications dans la zone acceptable [94 %, 96 %], le verdict est sans ambiguïté : le code est correct.

Cette validation est essentielle car elle isole les sources d'erreur dans la suite du projet. La Phase 3 pourra donc se concentrer sereinement sur les correctifs liés aux spécificités du dataset, en particulier le traitement des espèces dont la vraie classe est absente du fichier `ai_scores_all` du fait du seuil de troncature à 0,001.

## 7. Livrables

Le script Python est disponible dans `sanity_check/sanity_check_synthetic.py`. Les résultats détaillés par réplication sont sauvegardés dans `sanity_check_results.csv`. Les figures associées (distribution des coverages, coverage par réplication, distribution des tailles d'ensemble, visualisation 2D des données) se trouvent dans le dossier `figures_sanity/`.