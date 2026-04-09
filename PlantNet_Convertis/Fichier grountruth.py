import pandas as pd

# 1. Charger le fichier CSV
# Adapte le nom du fichier si nécessaire
file_path = "ground_truth.csv"
df = pd.read_csv(file_path)

# 2. Identifier les noms des colonnes
# Si le JSON a été converti simplement, les colonnes sont souvent l'index et la valeur
print("Colonnes détectées :", df.columns.tolist())

# On renomme pour plus de clarté (adapte selon tes noms de colonnes réels)
# Supposons : Col 1 = observation_id, Col 2 = ground_truth_val
df.columns = ['observation_id', 'ground_truth_val']

# 3. Filtrer les lignes différentes de "-1"
# On s'assure de traiter la valeur comme une chaîne de caractères (string)
df['ground_truth_val'] = df['ground_truth_val'].astype(str)

experts_df = df[df['ground_truth_val'] != "-1"]
non_experts_df = df[df['ground_truth_val'] == "-1"]

# 4. Afficher les statistiques pour ton rapport
total = len(df)
nb_experts = len(experts_df)
nb_non_experts = len(non_experts_df)

print(f"\n--- Statistiques Ground Truth ---")
print(f"Total d'observations : {total}")
print(f"Observations EXPERTES (différentes de -1) : {nb_experts} ({round(nb_experts/total*100, 2)}%)")
print(f"Observations NON-EXPERTES (valeur -1) : {nb_non_experts} ({round(nb_non_experts/total*100, 2)}%)")

# 5. Sauvegarder les experts dans un nouveau fichier pour le split
experts_df.to_csv("observations_experts.csv", index=False)
print("\nFichier 'observations_experts.csv' créé avec succès.")

# 6. Aperçu des premières lignes expertes
print("\nAperçu des experts :")
print(experts_df.head())