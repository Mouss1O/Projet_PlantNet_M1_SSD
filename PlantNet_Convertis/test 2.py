import json

import csv


# Chemins des fichiers

input_file = "ai_scores_all.json"

output_file = "ai_scores_all.csv"


print("Début de la conversion... Cela peut prendre un moment vu la taille du fichier.")


try:

    with open(input_file, 'r', encoding='utf-8') as f_json:

        # Charger le JSON en mémoire

        data = json.load(f_json)

        

    with open(output_file, 'w', newline='', encoding='utf-8') as f_csv:

        writer = csv.writer(f_csv)

        

        # Écrire l'en-tête (Header)

        writer.writerow(['observation_id', 'spicies_id', 'score'])

        

        # Parcourir le dictionnaire imbriqué
        # parent_id = "1006590000", children = {"42201": 0.88148, ...}

        for parent_id, children in data.items():

            for child_id, score in children.items():

                writer.writerow([parent_id, child_id, score])


    print(f"Succès ! Le fichier a été converti en : {output_file}")


except MemoryError:

    print("Erreur : Le fichier est trop gros pour être chargé d'un coup en mémoire.")

except Exception as e:

    print(f"Une erreur est survenue : {e}")
