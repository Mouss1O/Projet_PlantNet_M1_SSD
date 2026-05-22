"""
=============================================================================
Filtrage des observations expertes et non-expertes
=============================================================================
Projet : Pl@ntNet-CP — Prédiction Conformelle
Objectif : Séparer les observations validées par des experts
           (ground_truth_val != -1) des observations non expertisées
           (ground_truth_val == -1) à partir du fichier ground_truth.csv.

Entrées :
    - ground_truth.csv   (observation_id, ground_truth_val)

Sorties :
    - observations_experts.csv
    - observations_non_experts.csv
=============================================================================
"""

import pandas as pd
import os

# ================================================================
# CONFIGURATION
# ================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PATH_GT            = os.path.join(SCRIPT_DIR, "ground_truth.csv")
OUTPUT_EXPERTS     = os.path.join(SCRIPT_DIR, "observations_experts.csv")
OUTPUT_NON_EXPERTS = os.path.join(SCRIPT_DIR, "observations_non_experts.csv")

# ================================================================
# CHARGEMENT
# ================================================================
print("=" * 60)
print("FILTRAGE EXPERTS / NON-EXPERTS")
print("=" * 60)

print(f"\nChargement de {os.path.basename(PATH_GT)}...")
df = pd.read_csv(PATH_GT)
df.columns = ['observation_id', 'ground_truth_val']
df['ground_truth_val'] = df['ground_truth_val'].astype(str)

print(f"  Colonnes détectées : {df.columns.tolist()}")
print(f"  Observations totales : {len(df):,}")

# ================================================================
# FILTRAGE
# ================================================================
experts_df     = df[df['ground_truth_val'] != "-1"]
non_experts_df = df[df['ground_truth_val'] == "-1"]

# ================================================================
# STATISTIQUES RÉSUMÉES
# ================================================================
total          = len(df)
nb_experts     = len(experts_df)
nb_non_experts = len(non_experts_df)

n_especes      = experts_df['ground_truth_val'].nunique()
moy            = experts_df.groupby('ground_truth_val').size().mean()
med            = experts_df.groupby('ground_truth_val').size().median()
maxi           = experts_df.groupby('ground_truth_val').size().max()
mini           = experts_df.groupby('ground_truth_val').size().min()

print(f"\n--- Résumé du filtrage ---")
print(f"  Observations totales          : {total:,}")
print(f"  Observations expertes         : {nb_experts:,} ({100*nb_experts/total:.1f}%)")
print(f"  Observations non-expertes     : {nb_non_experts:,} ({100*nb_non_experts/total:.1f}%)")
print(f"\n--- Résumé des espèces expertes ---")
print(f"  Espèces uniques               : {n_especes:,}")
print(f"  Moyenne obs/espèce            : {moy:.1f}")
print(f"  Médiane obs/espèce            : {med:.0f}")
print(f"  Min obs/espèce                : {mini:,}")
print(f"  Max obs/espèce                : {maxi:,}")

# ================================================================
# SAUVEGARDE
# ================================================================
print(f"\nSauvegarde des fichiers...")

experts_df.to_csv(OUTPUT_EXPERTS, index=False)
print(f"  ✓ {os.path.basename(OUTPUT_EXPERTS)} créé ({nb_experts:,} lignes)")

non_experts_df.to_csv(OUTPUT_NON_EXPERTS, index=False)
print(f"  ✓ {os.path.basename(OUTPUT_NON_EXPERTS)} créé ({nb_non_experts:,} lignes)")

# ================================================================
# APERÇU
# ================================================================
print(f"\n--- Aperçu des observations expertes ---")
print(experts_df.head().to_string(index=False))

print(f"\n--- Aperçu des observations non-expertes ---")
print(non_experts_df.head().to_string(index=False))

# ================================================================
# TABLEAU RÉCAPITULATIF (pour le rapport)
# ================================================================
print(f"\n{'=' * 60}")
print("TABLEAU POUR LE RAPPORT")
print(f"{'=' * 60}")
print(f"| Catégorie        | Observations | Part    |")
print(f"|------------------|--------------|---------|")
print(f"| Expertes         | {nb_experts:>12,} | {100*nb_experts/total:>5.1f}%  |")
print(f"| Non-expertes     | {nb_non_experts:>12,} | {100*nb_non_experts/total:>5.1f}%  |")
print(f"| Total            | {total:>12,} | {'100.0':>5}%  |")

print(f"\n{'=' * 60}")
print("TERMINÉ")
print(f"{'=' * 60}")