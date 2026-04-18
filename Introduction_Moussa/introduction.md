# Introduction

La reconnaissance automatique des espèces végétales constitue un enjeu majeur pour la conservation de la biodiversité, la gestion des écosystèmes et l'éducation du grand public à la botanique. Dans ce domaine, le projet **Pl@ntNet** occupe une place de premier plan : son application mobile, basée sur la science participative, permet l'identification de plantes à partir de photographies et alimente en retour une base de données scientifique massive, exploitée par des centaines de chercheurs.

Notre travail s'appuie sur le jeu de données **Pl@ntNet-CrowdSWE-v2** (Lefort et al., 2024), qui rassemble les observations collectées entre 2017 et 2023 pour la région Europe du Sud-Ouest. L'exploration que nous avons menée révèle l'ampleur exacte des données à traiter : **5 561 512 observations** réparties en **81 783 751 lignes de scores** dans le fichier `ai_scores_all.csv`, couvrant **56 225 espèces distinctes** identifiées par l'IA. Parmi l'ensemble des observations, **21 624 ont été validées par des experts botanistes** (soit 0,39 % du total), constituant ainsi une référence fiable mais quantitativement limitée au regard de la diversité couverte.

Le modèle de deep learning sous-jacent à Pl@ntNet fournit en moyenne **14,7 scores par observation** (médiane = 10, maximum = 100), correspondant aux espèces les plus probables au-dessus d'un seuil de 0,001. Cependant, cette capacité prédictive globale masque une difficulté fondamentale mise en évidence par notre analyse descriptive : la **distribution en longue traîne** des observations. Dans notre dataset, **1 742 espèces (3,1 %) n'apparaissent qu'une seule fois** dans les prédictions du modèle, **39,5 % des espèces ont moins de 50 apparitions**, tandis que les **10 % d'espèces les plus fréquentes concentrent 79,7 % des prédictions**. Le **coefficient de Gini de 0,8636** traduit une inégalité extrême, comparable aux distributions les plus asymétriques observées en écologie.

Cette asymétrie a des conséquences directes sur la fiabilité du modèle : la **confiance médiane dans la prédiction top-1 n'est que de 0,695**, et dans **27,0 % des cas (1 502 199 observations)**, le score top-1 est inférieur à 0,5, signalant une incertitude importante sur plus d'un quart des prédictions. Seules **17,2 % des observations** bénéficient d'un score top-1 supérieur à 0,9.

Face à ce constat, la question centrale n'est plus seulement *« quelle est l'espèce prédite ? »* mais *« **avec quelle confiance ?** »*. Or, les modèles de classification standards produisent des scores softmax qui ne peuvent pas être interprétés directement comme des probabilités calibrées. Cela pose un problème concret : comment garantir à un utilisateur — qu'il soit botaniste, gestionnaire d'espaces naturels ou simple curieux — que la réponse fournie est fiable à un niveau donné ? Plutôt que de fournir une unique réponse avec une probabilité associée ou, à l'inverse, d'inonder l'utilisateur de suggestions, l'objectif est de construire un **ensemble de prédictions** dont la probabilité de contenir la bonne espèce atteint une garantie satisfaisante (typiquement 90 % ou 95 %), tout en restant le plus restreint possible.

C'est ici qu'intervient la **prédiction conforme** (*conformal prediction*), popularisée récemment par Angelopoulos et Bates (2022). Cette approche statistique transforme les scores bruts d'un classifieur en ensembles de prédiction assortis d'une garantie formelle de couverture :

$$
\mathbb{P}(Y \in \mathcal{C}(X)) \geq 1 - \alpha
$$

Autrement dit, pour un niveau d'erreur $\alpha$ fixé, l'ensemble retourné contient la vraie espèce avec une probabilité garantie, et ce **sans hypothèse paramétrique sur le modèle ni sur la distribution des données**, pourvu que l'hypothèse d'échangeabilité entre données de calibration et de test soit respectée.

L'objectif de ce projet est triple :

1. **Appliquer la prédiction conforme aux scores Pl@ntNet** sur le dataset CrowdSWE, en utilisant le score de non-conformité $s(x,y) = 1 - \hat{p}(y \mid x)$, et en évaluant la validité de la garantie de couverture sur les 21 624 observations expertes disponibles, divisées aléatoirement en 10 812 observations pour la calibration et 10 812 pour le test via un *ShuffleSplit* 50/50.

2. **Évaluer la validité des garanties** au niveau marginal global, mais surtout au niveau conditionnel par espèce, en portant une attention particulière aux espèces rares — celles pour lesquelles la garantie théorique est la plus fragile. Nous avons notamment identifié **624 espèces présentes dans le test mais absentes de la calibration**, qui posent un défi fondamental à l'approche standard.

3. **Explorer les extensions récentes de la méthode** : la technique **PAS** (Ding, Fermanian & Salmon, 2025) conçue spécifiquement pour corriger les défaillances de la prédiction conforme sur les distributions en longue traîne, ainsi que l'impact du **temperature scaling** (Dabah & Tirer, 2024) sur la calibration des scores softmax et la taille des ensembles de prédiction.

Une contribution importante de ce travail réside dans l'**analyse critique des hypothèses** sous-jacentes à la prédiction conforme dans un contexte réel. Le dataset CrowdSWE présente en effet plusieurs caractéristiques susceptibles de remettre en question l'échangeabilité : couverture taxonomique partielle (limitée à l'Europe du Sud-Ouest), biais de contribution, et hétérogénéité de la qualité des labels crowdsourcés. Nous discuterons dans quelle mesure ces spécificités affectent la validité des garanties obtenues et quels prétraitements peuvent atténuer ces biais.

Le reste de ce rapport s'organise comme suit. La **section 2** présente le dataset Pl@ntNet-CrowdSWE-v2 et en propose une analyse statistique descriptive détaillée. La **section 3** introduit le cadre théorique de la prédiction conforme. La **section 4** détaille notre méthodologie expérimentale. La **section 5** présente les résultats obtenus, incluant l'évaluation par strates de rareté. La **section 6** aborde les pistes d'amélioration via PAS et le temperature scaling. Enfin, la **section 7** discute les limites de notre approche et ouvre sur des perspectives.
