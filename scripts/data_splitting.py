"""
data_splitting.py
ShuffleSplit 50/50 sur les observations expertes
"""

import pandas as pd
import os
import sys
from sklearn.model_selection import ShuffleSplit

# ================================================================
# CHEMINS RELATIFS
# ================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.dirname(SCRIPT_DIR)

DIR_VOTES = os.path.join(ROOT, "Données", "votes")
DIR_OUT   = os.path.join(ROOT, "data", "split_random")

PATH_EXPERTS = os.path.join(DIR_VOTES, "observations_experts.csv")
PATH_CAL     = os.path.join(DIR_OUT,   "calibration.csv")
PATH_TEST    = os.path.join(DIR_OUT,   "test.csv")


def main():
    print("="*55)
    print("SHUFFLESPLIT 50/50")
    print("="*55)
    print(f"Racine projet : {ROOT}")
    
    if not os.path.exists(PATH_EXPERTS):
        raise FileNotFoundError(
            f"Fichier manquant : {PATH_EXPERTS}\n"
            f"Lancez d'abord observations_expert.py"
        )
    
    os.makedirs(DIR_OUT, exist_ok=True)
    
    try:
        df = pd.read_csv(PATH_EXPERTS)
    except pd.errors.EmptyDataError:
        raise ValueError(f"Fichier vide : {PATH_EXPERTS}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Erreur parsing CSV : {e}")
    
    required = {"observation_id", "ground_truth_val"}
    missing  = required - set(df.columns)
    if missing:
        raise KeyError(f"Colonnes manquantes : {missing}")
    
    df["ground_truth_val"] = pd.to_numeric(
        df["ground_truth_val"], errors="coerce"
    ).fillna(-1).astype(int)
    df = df[df["ground_truth_val"] >= 0].reset_index(drop=True)
    
    if len(df) == 0:
        raise ValueError("Aucune observation experte valide")
    
    print(f"Observations expertes : {len(df):,}")
    print(f"Especes uniques       : {df['ground_truth_val'].nunique():,}")
    
    ss = ShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    idx_cal, idx_test = next(ss.split(df))
    
    cal_df  = df.iloc[idx_cal].reset_index(drop=True)
    test_df = df.iloc[idx_test].reset_index(drop=True)
    
    print(f"\n=== Split ===")
    print(f"Calibration (50%) : {len(cal_df):>6,}")
    print(f"Test        (50%) : {len(test_df):>6,}")
    
    print(f"\n=== Verifications ===")
    
    n_dup_cal  = cal_df["observation_id"].duplicated().sum()
    n_dup_test = test_df["observation_id"].duplicated().sum()
    
    status_cal  = "OK" if n_dup_cal == 0 else "ERREUR"
    status_test = "OK" if n_dup_test == 0 else "ERREUR"
    
    print(f"Doublons dans cal  : {n_dup_cal}   [{status_cal}]")
    print(f"Doublons dans test : {n_dup_test}  [{status_test}]")
    
    overlap = set(cal_df["observation_id"]) & set(test_df["observation_id"])
    status_overlap = "OK" if len(overlap) == 0 else "ERREUR"
    print(f"Overlap cal-test   : {len(overlap)}   [{status_overlap}]")
    
    if overlap:
        raise ValueError(f"Fuite detectee : {len(overlap)} obs dans cal ET test")
    
    all_ids = pd.concat([cal_df["observation_id"], test_df["observation_id"]])
    n_unique = all_ids.nunique()
    n_total  = len(all_ids)
    status_unique = "OK" if n_unique == n_total else "ERREUR"
    print(f"IDs uniques global : {n_unique:,} / {n_total:,}   [{status_unique}]")
    
    print(f"\n=== Distribution especes ===")
    for name, sub in [("Calibration", cal_df), ("Test", test_df)]:
        c = sub["ground_truth_val"].value_counts()
        print(f"\n  {name}")
        print(f"    Especes uniques       : {c.shape[0]:,}")
        print(f"    Especes avec 1 obs    : {(c == 1).sum():,}")
        print(f"    Especes avec < 5 obs  : {(c < 5).sum():,}")
        print(f"    Especes avec >= 10    : {(c >= 10).sum():,}")
    
    sp_cal  = set(cal_df["ground_truth_val"])
    sp_test = set(test_df["ground_truth_val"])
    orphans = sp_test - sp_cal
    print(f"\nATTENTION : especes dans test absentes du cal : {len(orphans):,}")
    
    try:
        cal_df.to_csv(PATH_CAL,   index=False)
        test_df.to_csv(PATH_TEST, index=False)
    except PermissionError as e:
        raise PermissionError(f"Ecriture refusee : {e}")
    except OSError as e:
        raise OSError(f"Erreur I/O : {e}")
    
    print(f"\nFichiers crees :")
    print(f"   {PATH_CAL}")
    print(f"   {PATH_TEST}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrompu")
        sys.exit(130)
    except Exception as e:
        print(f"\nERREUR : {e}")
        sys.exit(1)