"""
observations_expert.py
Separe experts (ground_truth >= 0) et non-experts (-1)
"""

import pandas as pd
import os
import sys

# ================================================================
# CHEMINS RELATIFS
# ================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.dirname(SCRIPT_DIR)

DIR_VOTES = os.path.join(ROOT, "Données", "votes")

PATH_GT       = os.path.join(DIR_VOTES, "ground_truth.csv")
PATH_EXPERTS  = os.path.join(DIR_VOTES, "observations_experts.csv")
PATH_NONEXP   = os.path.join(DIR_VOTES, "observations_non_experts.csv")


def main():
    print("="*55)
    print("SEPARATION EXPERTS / NON-EXPERTS")
    print("="*55)
    print(f"Racine projet : {ROOT}")
    
    if not os.path.exists(PATH_GT):
        raise FileNotFoundError(
            f"Fichier manquant : {PATH_GT}\n"
            f"Lancez d'abord convert_json.py"
        )
    
    try:
        df = pd.read_csv(PATH_GT)
    except pd.errors.EmptyDataError:
        raise ValueError(f"Fichier vide : {PATH_GT}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Erreur parsing CSV : {e}")
    
    required = {"observation_id", "ground_truth_val"}
    missing  = required - set(df.columns)
    if missing:
        raise KeyError(f"Colonnes manquantes : {missing}")
    
    print(f"Observations chargees : {len(df):,}")
    
    df["ground_truth_val"] = pd.to_numeric(
        df["ground_truth_val"], errors="coerce"
    ).fillna(-1).astype(int)
    
    n_dup = df["observation_id"].duplicated().sum()
    if n_dup > 0:
        print(f"ATTENTION : {n_dup} doublons detectes - suppression")
        df = df.drop_duplicates(subset="observation_id").reset_index(drop=True)
    
    experts_df     = df[df["ground_truth_val"] >= 0].copy().reset_index(drop=True)
    non_experts_df = df[df["ground_truth_val"] == -1].copy().reset_index(drop=True)
    
    total = len(df)
    n_exp = len(experts_df)
    n_ne  = len(non_experts_df)
    
    print(f"\n=== Separation ===")
    print(f"Experts    (>= 0) : {n_exp:>9,}  ({100*n_exp/total:.2f}%)")
    print(f"Non-experts (-1)  : {n_ne:>9,}  ({100*n_ne/total:.2f}%)")
    
    species_counts = experts_df["ground_truth_val"].value_counts()
    print(f"\n=== Distribution especes (experts) ===")
    print(f"Especes uniques        : {len(species_counts):>6,}")
    print(f"Mediane obs/espece     : {species_counts.median():>6.0f}")
    print(f"Moyenne obs/espece     : {species_counts.mean():>6.1f}")
    print(f"Max obs/espece         : {species_counts.max():>6,}")
    print(f"Especes avec 1 obs     : {(species_counts == 1).sum():>6,}")
    print(f"Especes avec < 5 obs   : {(species_counts < 5).sum():>6,}")
    print(f"Especes avec < 10 obs  : {(species_counts < 10).sum():>6,}")
    print(f"Especes avec >= 50 obs : {(species_counts >= 50).sum():>6,}")
    
    if n_exp == 0:
        raise ValueError("Aucun expert trouve dans les donnees")
    
    if n_exp + n_ne != total:
        raise ValueError(f"Incoherence : {n_exp} + {n_ne} != {total}")
    
    try:
        experts_df.to_csv(PATH_EXPERTS, index=False)
        non_experts_df.to_csv(PATH_NONEXP, index=False)
    except PermissionError as e:
        raise PermissionError(f"Ecriture refusee : {e}")
    except OSError as e:
        raise OSError(f"Erreur I/O : {e}")
    
    print(f"\nFichiers crees :")
    print(f"   {PATH_EXPERTS}")
    print(f"   {PATH_NONEXP}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrompu")
        sys.exit(130)
    except Exception as e:
        print(f"\nERREUR : {e}")
        sys.exit(1)