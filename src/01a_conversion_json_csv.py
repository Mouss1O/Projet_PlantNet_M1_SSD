"""
=============================================================================
Conversion JSON → CSV
=============================================================================
Projet : Pl@ntNet-CP — Prédiction Conformelle
Objectif : Convertir les fichiers de scores IA et de vérité terrain
           du format JSON vers le format CSV.

Entrées :
    - ai_scores_all.json   (observation_id → {species_id → score})
    - ground_truth.json    (observation_id → ground_truth_val)

Sorties :
    - ai_scores_all.csv    (observation_id, species_id, score)
    - ground_truth.csv     (observation_id, ground_truth_val)
=============================================================================
"""

import json
import csv
import os
import pandas as pd

# ================================================================
# CONFIGURATION
# ================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PATH_AI_SCORES   = os.path.join(SCRIPT_DIR, "ai_scores_all.json")
PATH_GT          = os.path.join(SCRIPT_DIR, "ground_truth.json")

OUTPUT_AI_SCORES = os.path.join(SCRIPT_DIR, "ai_scores_all.csv")
OUTPUT_GT        = os.path.join(SCRIPT_DIR, "ground_truth.csv")

BATCH_SIZE = 10_000  # Nombre d'observations traitées avant écriture

# ================================================================
# CONVERSION : ai_scores_all.json
# Structure : {observation_id: {species_id: score, ...}, ...}
# Traitement par batch pour éviter la saturation RAM
# ================================================================
print("=" * 60)
print("CONVERSION JSON → CSV")
print("=" * 60)

print(f"\nTraitement de ai_scores_all.json...")
try:
    n_obs_total  = 0
    n_rows_total = 0
    first_write  = True
    batch        = []

    with open(PATH_AI_SCORES, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(OUTPUT_AI_SCORES, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["observation_id", "species_id", "score"])

        for obs_id, species_scores in data.items():
            for species_id, score in species_scores.items():
                batch.append([obs_id, species_id, score])
                n_rows_total += 1

            n_obs_total += 1

            # Écriture par batch pour libérer la RAM
            if n_obs_total % BATCH_SIZE == 0:
                writer.writerows(batch)
                batch = []
                print(f"  Observations traitées : {n_obs_total:,} — Lignes écrites : {n_rows_total:,}")

        # Écriture du dernier batch
        if batch:
            writer.writerows(batch)

    print(f"  Fichier créé              : ai_scores_all.csv")
    print(f"  Observations traitées     : {n_obs_total:,}")
    print(f"  Lignes totales            : {n_rows_total:,}")
    print(f"  ✓ Conversion réussie")

except FileNotFoundError:
    print(f"  ✗ Fichier introuvable : ai_scores_all.json")
    print(f"    → Vérifiez que le fichier est bien dans le même dossier que ce script.")
except Exception as e:
    print(f"  ✗ Erreur : {e}")

# ================================================================
# CONVERSION : ground_truth.json
# Structure : {observation_id: ground_truth_val}
# ================================================================
print(f"\nTraitement de ground_truth.json...")
try:
    with open(PATH_GT, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df_gt = pd.DataFrame(list(data.items()), columns=["observation_id", "ground_truth_val"])
    df_gt["ground_truth_val"] = pd.to_numeric(df_gt["ground_truth_val"])
    df_gt.to_csv(OUTPUT_GT, index=False)

    n_obs  = len(df_gt)
    n_ids  = df_gt["observation_id"].nunique()
    n_neg  = (df_gt["ground_truth_val"] < 0).sum()
    n_pos  = (df_gt["ground_truth_val"] > 0).sum()
    mini   = df_gt["ground_truth_val"].min()
    maxi   = df_gt["ground_truth_val"].max()

    print(f"  Fichier créé              : ground_truth.csv")
    print(f"  Observations              : {n_obs:,}")
    print(f"  IDs uniques               : {n_ids:,}")
    print(f"  Valeurs négatives (-1)    : {n_neg:,} ({100*n_neg/n_obs:.1f}%)")
    print(f"  Valeurs positives         : {n_pos:,} ({100*n_pos/n_obs:.1f}%)")
    print(f"  Min                       : {mini}")
    print(f"  Max                       : {maxi}")
    print(f"  ✓ Conversion réussie")

except FileNotFoundError:
    print(f"  ✗ Fichier introuvable : ground_truth.json")
    print(f"    → Vérifiez que le fichier est bien dans le même dossier que ce script.")
except Exception as e:
    print(f"  ✗ Erreur : {e}")

# ================================================================
# TABLEAU RÉCAPITULATIF (pour le rapport)
# ================================================================
print(f"\n{'=' * 60}")
print("TABLEAU POUR LE RAPPORT")
print(f"{'=' * 60}")
print(f"| Fichier source      | Fichier produit      | Lignes       |")
print(f"|---------------------|----------------------|--------------|")
for src, out in [("ai_scores_all.json", OUTPUT_AI_SCORES), ("ground_truth.json", OUTPUT_GT)]:
    try:
        n = sum(1 for _ in open(out)) - 1  # Compte les lignes sans charger en RAM
        print(f"| {src:<19} | {os.path.basename(out):<20} | {n:>12,} |")
    except Exception:
        print(f"| {src:<19} | {os.path.basename(out):<20} | {'N/A':>12} |")

print(f"\n{'=' * 60}")
print("TERMINÉ")
print(f"{'=' * 60}")