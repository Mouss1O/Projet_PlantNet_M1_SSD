# test.py
import pandas as pd
import json
import os
import glob

# Préparation et Fusion des données

# Chargement des fichiers
ai_scores_all = pd.read_csv("ai_scores_all.csv")








# Calcul de la Prévalence (La Longue Traîne)

import pandas as pd
import matplotlib.pyplot as plt

# --- Calcul du nombre d'images par espèce ---

prevalence = (
    ai_scores_all
        .groupby("spicies_id")
        .size()
        .reset_index(name="n_images")
        .sort_values("n_images", ascending=False)
        .reset_index(drop=True)
)

# Création du rank (équivalent row_number())
prevalence["rank"] = prevalence.index + 1

# Affichage des premières lignes
print(prevalence.head())


# --- Visualisation Longue Traîne ---

plt.figure()
plt.fill_between(prevalence["rank"], prevalence["n_images"], alpha=0.5)
plt.plot(prevalence["rank"], prevalence["n_images"])

plt.yscale("log")

plt.title("Distribution de la Prévalence (Long Tail)")
plt.xlabel("Rang de l'espèce (de la plus commune à la plus rare)")
plt.ylabel("Nombre d'observations")

plt.show()




summary_stats = {
    "total_especes": prevalence.shape[0],
    "moyenne_images": prevalence["n_images"].mean(),
    "mediane_images": prevalence["n_images"].median(),
    "nb_especes_rares": (prevalence["n_images"] < 5).sum()  # Moins de 5 images
}

print(summary_stats)



