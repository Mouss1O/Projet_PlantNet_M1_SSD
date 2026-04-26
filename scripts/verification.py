import pandas as pd

# Vérifier une observation au hasard
scores = pd.read_csv("fichiers_utilises/ai_scores_all.csv")
print("Aperçu des scores :")
print(scores.head(10))

print("\nStatistiques des scores :")
print(scores['score'].describe())