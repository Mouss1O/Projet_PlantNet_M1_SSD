"""
convert_json.py - Conversion JSON vers CSV avec streaming (ijson)
"""

import json
import csv
import os
import sys
import ijson
from collections import Counter

# ================================================================
# CHEMINS RELATIFS (portables entre machines)
# ================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.dirname(SCRIPT_DIR)

BASE       = os.path.join(ROOT, "Données")
DIR_SCORES = os.path.join(BASE, "ai_scores")
DIR_VOTES  = os.path.join(BASE, "votes")

# ================================================================
class ConversionError(Exception): pass
class InvalidJSONStructureError(ConversionError): pass

# ================================================================
def convert_ai_scores_streaming(json_path, csv_path):
    print(f"\n=== {os.path.basename(json_path)} ===")
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Introuvable : {json_path}")
    
    size_mb = os.path.getsize(json_path) / 1e6
    print(f"  Taille    : {size_mb:.1f} MB (streaming)")
    
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    n_rows = 0
    n_obs  = 0
    n_skip = 0
    
    try:
        with open(json_path, 'rb') as f_json, \
             open(csv_path, 'w', newline='', encoding='utf-8') as f_csv:
            
            writer = csv.writer(f_csv)
            writer.writerow(['observation_id', 'spicies_id', 'score'])
            
            for obs_id, scores_dict in ijson.kvitems(f_json, ''):
                if not isinstance(scores_dict, dict):
                    n_skip += 1
                    continue
                
                for sp_id, score in scores_dict.items():
                    try:
                        writer.writerow([obs_id, int(sp_id), float(score)])
                        n_rows += 1
                    except (ValueError, TypeError):
                        n_skip += 1
                
                n_obs += 1
                if n_obs % 10000 == 0:
                    print(f"  ... {n_obs:,} observations traitees")
    
    except ijson.JSONError as e:
        raise ConversionError(f"JSON mal forme : {e}")
    except PermissionError as e:
        raise ConversionError(f"Permission refusee : {e}")
    
    print(f"  Observations    : {n_obs:,}")
    print(f"  Lignes ecrites  : {n_rows:,}")
    if n_skip > 0:
        print(f"  ATTENTION : Skipped = {n_skip}")
    print(f"  -> {csv_path}")

# ================================================================
def convert_ground_truth(json_path, csv_path):
    print(f"\n=== {os.path.basename(json_path)} ===")
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Introuvable : {json_path}")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ConversionError(f"JSON mal forme : {e}")
    
    if not isinstance(data, dict) or len(data) == 0:
        raise InvalidJSONStructureError("Structure invalide")
    
    print(f"  Entrees : {len(data):,}")
    
    first_val = next(iter(data.values()))
    is_list   = isinstance(first_val, list)
    print(f"  Structure : {'liste (MV)' if is_list else 'valeur unique'}")
    
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    n_experts = n_minus1 = n_ties = n_errors = 0
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['observation_id', 'ground_truth_val'])
        
        for obs_id, val in data.items():
            try:
                if is_list:
                    if not val:
                        label = -1
                    else:
                        ranking = Counter(val).most_common()
                        if len(ranking) >= 2 and ranking[0][1] == ranking[1][1]:
                            label = -1
                            n_ties += 1
                        else:
                            label = int(ranking[0][0])
                else:
                    label = int(val)
            except (ValueError, TypeError):
                n_errors += 1
                label = -1
            
            if label == -1:
                n_minus1 += 1
            else:
                n_experts += 1
            
            writer.writerow([obs_id, label])
    
    print(f"  Experts     : {n_experts:,}")
    print(f"  Non-experts : {n_minus1:,}")
    if n_ties:   print(f"  Egalites MV : {n_ties:,}")
    if n_errors: print(f"  ATTENTION : Erreurs = {n_errors}")
    print(f"  -> {csv_path}")
    
    if n_experts == 0:
        raise ConversionError("Aucun expert detecte")

# ================================================================
def inspect_json_streaming(json_path, n_samples=3):
    print(f"\n=== Inspection {os.path.basename(json_path)} ===")
    
    if not os.path.exists(json_path):
        print(f"  ATTENTION : Absent : {json_path}")
        return
    
    try:
        with open(json_path, 'rb') as f:
            count = 0
            for obs_id, val in ijson.kvitems(f, ''):
                if count < n_samples:
                    print(f"  {obs_id} -> {type(val).__name__}")
                    print(f"    Exemple : {str(val)[:150]}")
                count += 1
                if count >= 1000:
                    break
            print(f"  Entrees (min.) : {count:,}")
    except Exception as e:
        print(f"  ATTENTION : Erreur inspection : {e}")

# ================================================================
def main():
    print("="*60)
    print("CONVERSION JSON VERS CSV")
    print("="*60)
    print(f"Racine projet : {ROOT}")
    
    for d in [DIR_SCORES, DIR_VOTES]:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Dossier introuvable : {d}")
    
    try:
        convert_ai_scores_streaming(
            os.path.join(DIR_SCORES, "ai_scores_all.json"),
            os.path.join(DIR_SCORES, "ai_scores_all.csv")
        )
    except Exception as e:
        print(f"\nCRITIQUE ai_scores : {e}")
        sys.exit(1)
    
    try:
        convert_ground_truth(
            os.path.join(DIR_VOTES, "ground_truth.json"),
            os.path.join(DIR_VOTES, "ground_truth.csv")
        )
    except Exception as e:
        print(f"\nCRITIQUE ground_truth : {e}")
        sys.exit(1)
    
    for fname in ["ai_votes.json", "human_votes.json", "PN_valid_votes.json"]:
        try:
            inspect_json_streaming(os.path.join(DIR_VOTES, fname))
        except Exception as e:
            print(f"  ATTENTION : {fname} : {e}")
    
    print(f"\n{'='*60}")
    print("TERMINE")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrompu")
        sys.exit(130)
    except Exception as e:
        print(f"\nERREUR FATALE : {e}")
        sys.exit(1)