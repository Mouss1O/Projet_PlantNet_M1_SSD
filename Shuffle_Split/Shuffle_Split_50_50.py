"""
=============================================================================
Construction des jeux de calibration et de test (50% / 50%)
=============================================================================
Projet : Pl@ntNet-CP — Prédiction Conformelle
Objectif : Diviser les observations expertes en deux jeux équilibrés
           et extraire les scores IA correspondants depuis le fichier
           complet ai_scores_all.csv.

Entrées :
    - observations_experts.csv   (observation_id, true_species_id)
    - ai_scores_all.csv          (observation_id, spicies_id, ...)

Sorties :
    - expert_calib_50.csv
    - expert_test_50.csv
=============================================================================
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import os

# ================================================================
# CONFIGURATION
# ================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PATH_EXPERTS   = os.path.join(SCRIPT_DIR, "observations_experts.csv")
PATH_AI_SCORES = os.path.join(SCRIPT_DIR, "ai_scores_all.csv")

OUTPUT_CALIB   = os.path.join(SCRIPT_DIR, "expert_calib_50.csv")
OUTPUT_TEST    = os.path.join(SCRIPT_DIR, "expert_test_50.csv")

CHUNK_SIZE     = 5_000_000
TEST_SIZE      = 0.50
RANDOM_STATE   = 42

# ================================================================
# CHARGEMENT DES OBSERVATIONS EXPERTES
# ================================================================
print("=" * 60)
print("CONSTRUCTION CALIBRATION / TEST (50% / 50%)")
print("=" * 60)

print(f"\nChargement de {os.path.basename(PATH_EXPERTS)}...")
df_experts = pd.read_csv(PATH_EXPERTS)
df_experts.columns = ['observation_id', 'true_species_id']
df_experts['observation_id']  = df_experts['observation_id'].astype(str)
df_experts['true_species_id'] = df_experts['true_species_id'].astype(str)

print(f"  Observations expertes totales : {len(df_experts):,}")

# ================================================================
# SHUFFLE SPLIT 50% / 50%
# ================================================================
df_calib_ids, df_test_ids = train_test_split(
    df_experts,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print(f"\n--- Répartition du split ---")
print(f"  Calibration : {len(df_calib_ids):,} observations ({100*(1-TEST_SIZE):.0f}%)")
print(f"  Test        : {len(df_test_ids):,} observations ({100*TEST_SIZE:.0f}%)")

# ================================================================
# EXTRACTION PAR CHUNKS
# ================================================================
def extract_scores(ids_df, output_path):
    """
    Parcourt ai_scores_all.csv par chunks et extrait les lignes
    dont (observation_id, spicies_id) correspondent aux experts
    du jeu ids_df.
    """
    output_name = os.path.basename(output_path)
    print(f"\nExtraction des scores pour {output_name}...")

    first_write  = True
    n_rows_total = 0
    n_chunks     = 0

    for chunk in pd.read_csv(PATH_AI_SCORES, chunksize=CHUNK_SIZE):
        chunk['observation_id'] = chunk['observation_id'].astype(str)
        chunk['species_id']     = chunk['species_id'].astype(str)
        n_chunks += 1

        match = pd.merge(
            chunk, ids_df,
            left_on=['observation_id', 'species_id'],
            right_on=['observation_id', 'true_species_id']
        )

        if not match.empty:
            match.to_csv(
                output_path,
                index=False,
                mode='a' if not first_write else 'w',
                header=first_write
            )
            first_write   = False
            n_rows_total += len(match)

        print(f"  Chunk {n_chunks:>3} traité — lignes extraites jusqu'ici : {n_rows_total:,}")

    print(f"  ✓ {output_name} terminé : {n_rows_total:,} lignes extraites")
    return n_rows_total

# ================================================================
# EXÉCUTION
# ================================================================
n_calib = extract_scores(df_calib_ids, OUTPUT_CALIB)
n_test  = extract_scores(df_test_ids,  OUTPUT_TEST)

# ================================================================
# TABLEAU RÉCAPITULATIF (pour le rapport)
# ================================================================
print(f"\n{'=' * 60}")
print("TABLEAU POUR LE RAPPORT")
print(f"{'=' * 60}")
print(f"| Jeu         | Observations | Lignes extraites |")
print(f"|-------------|--------------|------------------|")
print(f"| Calibration | {len(df_calib_ids):>12,} | {n_calib:>16,} |")
print(f"| Test        | {len(df_test_ids):>12,} | {n_test:>16,} |")
print(f"| Total       | {len(df_experts):>12,} | {n_calib+n_test:>16,} |")

print(f"\n{'=' * 60}")
print("TERMINÉ")
print(f"{'=' * 60}")