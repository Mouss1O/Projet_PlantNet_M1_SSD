# Prédiction conformelle et base de données Pl@ntNet-CrowdSWE

## Présentation du projet
Ce projet s'inscrit dans le cadre du Master 1 Statistique et Science des Données (SSD) à l'Université de Montpellier. L'objectif est d'appliquer les méthodes de **prédiction conformelle (Conformal Prediction)** aux données issues de l'application Pl@ntNet afin de quantifier l'incertitude des prédictions d'identification d'espèces végétales.

Un enjeu majeur de ce travail est de traiter la problématique de la **longue traîne (long-tail)**, où de nombreuses espèces rares disposent de peu d'images d'apprentissage, rendant leur identification incertaine.

## Membres du groupe
- Dossou AGOSSOU
- Firdaousse KARIMOU
- Moussa DIAGNE

**Encadrants :** Joseph Salmon, Christophe Botella, Jean-Baptiste Fermanian.

## Objectifs principaux
1. **Exploration des données :** Analyse de la base Pl@ntNet-CrowdSWE (distribution des espèces, prévalence).
2. **Calibration :** Mise en œuvre de la méthode de prédiction conformelle marginale (approche Angelopoulos et al.).
3. **Optimisation :** Étude et comparaison avec la méthode **PAS (Partitioned Adaptive Set)** pour améliorer la couverture des espèces rares.
4. **Impact de la température :** Analyse de l'ajustement du score Softmax sur les performances de calibration.

## Données utilisées
Les données proviennent du dépôt Zenodo : [Pl@ntNet-CrowdSWE](https://zenodo.org/records/17913995).
Fichiers principaux en cours de traitement :
- `ai_votes.json` (Données d'IA et votes)
- `ai_scores.json` & `ai_scores_all.json`

## Installation et Environnement
Le projet utilise principalement **Python** (ou R) et l'IDE **Positron**.