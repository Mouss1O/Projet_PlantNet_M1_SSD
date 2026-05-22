"""
=============================================================================
Construction des jeux complets (Multi-labels)
=============================================================================
Projet : Pl@ntNet-CP
Objectif : Extraire TOUS les scores (top-K) pour les observations expertes.
           Modification cruciale : Jointure sur observation_id uniquement.
=============================================================================
"""

import pandas as pd
import os

# CONFIGURATION
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_EXPERTS   = os.path.join(SCRIPT_DIR, "observations_experts.csv")
PATH_AI_SCORES = os.path.join(SCRIPT_DIR, "ai_scores_all.csv")
OUTPUT_CALIB   = os.path.join(SCRIPT_DIR, "expert_calib_50.csv")
OUTPUT_TEST    = os.path.join(SCRIPT_DIR, "expert_test_50.csv")

CHUNK_SIZE = 5_000_000

# 1. CHARGEMENT DES IDS (On garde le même split pour la cohérence)
df_experts = pd.read_csv(PATH_EXPERTS)
df_experts.columns = ['observation_id', 'true_species_id']
df_experts['observation_id'] = df_experts['observation_id'].astype(str)

# Ici, refais ton split 50/50 comme avant
from sklearn.model_selection import train_test_split
df_calib_ids, df_test_ids = train_test_split(df_experts, test_size=0.5, random_state=42)

def extract_all_labels(ids_df, output_path):
    print(f"\nExtraction de tous les labels pour {os.path.basename(output_path)}...")
    first_write = True
    n_obs_found = 0
    
    for chunk in pd.read_csv(PATH_AI_SCORES, chunksize=CHUNK_SIZE):
        chunk['observation_id'] = chunk['observation_id'].astype(str)
        
        # JOINTURE CRUCIALE : Uniquement sur observation_id
        # Cela ramène TOUTES les espèces proposées par l'IA pour ces images
        match = pd.merge(chunk, ids_df, on='observation_id', how='inner')
        
        if not match.empty:
            match.to_csv(output_path, index=False, mode='a' if not first_write else 'w', header=first_write)
            first_write = False
            n_obs_found = match['observation_id'].nunique()
            print(f"  Observations traitées : {n_obs_found:,}")

    print(f"✓ Terminé.")

# EXÉCUTION
extract_all_labels(df_calib_ids, OUTPUT_CALIB)
extract_all_labels(df_test_ids, OUTPUT_TEST)