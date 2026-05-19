"""
=============================================================================
CP Marginale Naïve — Standard + PAS (Sans correction de biais)
=============================================================================
Projet : Pl@ntNet-CP — Prédiction Conformelle
Blog   : J. Salmon — Conformal Prediction

Scores :
    Standard : s(x,y) = -p̂(y|x)
    PAS      : s(x,y) = -p̂(y|x) / p̂(y)

Quantile : q̂ = quantile de niveau ⌈(n+1)(1-α)⌉ / n

Sorties  : tableau récapitulatif + figures
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from sklearn.model_selection import train_test_split

def log(msg=""):
    print(msg, flush=True)

# ================================================================
# CONFIGURATION
# ================================================================
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)  # racine du dépôt

PATH_EXPERTS   = os.path.join(PROJECT_DIR, "Expert_NonExpert",   "observations_experts.csv")
PATH_AI_SCORES = os.path.join(PROJECT_DIR, "convertir_json_csv", "ai_scores_all.csv")
DIR_FIGURES    = os.path.join(SCRIPT_DIR, "figures")
DIR_SORTIES    = os.path.join(SCRIPT_DIR, "sorties")
os.makedirs(DIR_FIGURES, exist_ok=True)
os.makedirs(DIR_SORTIES, exist_ok=True)

ALPHAS       = [0.01, 0.05, 0.10, 0.20]
TEST_SIZE    = 0.50
RANDOM_STATE = 42
CHUNK_SIZE   = 500_000

# ================================================================
# ÉTAPE 1 : CHARGEMENT ET SPLIT 50/50
# ================================================================
log("=" * 70)
log("CP MARGINALE NAÏVE — STANDARD + PAS")
log("=" * 70)

log(f"\nChargement de observations_experts.csv...")
df_experts = pd.read_csv(PATH_EXPERTS)
df_experts.columns = ['observation_id', 'true_species_id']
df_experts['observation_id']  = df_experts['observation_id'].astype(str)
df_experts['true_species_id'] = df_experts['true_species_id'].astype(str)

df_calib, df_test = train_test_split(
    df_experts, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
df_calib = df_calib.reset_index(drop=True)
df_test  = df_test.reset_index(drop=True)
n_cal  = len(df_calib)
n_test = len(df_test)
log(f"  Calibration : {n_cal:,}")
log(f"  Test        : {n_test:,}")

calib_id_set = set(df_calib['observation_id'])
test_id_set  = set(df_test['observation_id'])

# ================================================================
# ÉTAPE 2 : LECTURE DES SCORES IA PAR CHUNKS
# ================================================================
log(f"\nLecture de ai_scores_all.csv par chunks de {CHUNK_SIZE:,}...")
calib_chunks, test_chunks = [], []

AI_DTYPES = {'score': 'float32'}
for i, chunk in enumerate(pd.read_csv(PATH_AI_SCORES, chunksize=CHUNK_SIZE, dtype=AI_DTYPES), 1):
    chunk['observation_id'] = chunk['observation_id'].astype(str)
    chunk['species_id']     = chunk['species_id'].astype(str)

    cal_mask = chunk['observation_id'].isin(calib_id_set)
    tst_mask = chunk['observation_id'].isin(test_id_set)

    if cal_mask.any():
        calib_chunks.append(chunk.loc[cal_mask, ['observation_id', 'species_id', 'score']].copy())
    if tst_mask.any():
        test_chunks.append(chunk.loc[tst_mask, ['observation_id', 'species_id', 'score']].copy())

    del chunk
    log(f"  Chunk {i} traité")

df_cal_scores  = pd.concat(calib_chunks, ignore_index=True)
df_test_scores = pd.concat(test_chunks, ignore_index=True)
log(f"  Scores cal  : {len(df_cal_scores):,} lignes")
log(f"  Scores test : {len(df_test_scores):,} lignes")

# ================================================================
# ÉTAPE 3 : SCORES DE NON-CONFORMITÉ (CALIBRATION)
# ================================================================
log(f"\nCalcul des scores de non-conformité...")

# 3a. Score de la vraie espèce pour chaque obs cal
df_cal_true = df_cal_scores.merge(
    df_calib,
    left_on=['observation_id', 'species_id'],
    right_on=['observation_id', 'true_species_id'],
    how='inner'
)
matched_cal_ids = set(df_cal_true['observation_id'])
missing_cal_ids = calib_id_set - matched_cal_ids
n_missing = len(missing_cal_ids)

log(f"  Vraie espèce dans top-k  : {len(matched_cal_ids):,}")
log(f"  Vraie espèce ABSENTE     : {n_missing:,} ({100*n_missing/n_cal:.1f}%)")

# 3b. Prévalence p̂(y) depuis la calibration
species_counts = df_calib['true_species_id'].value_counts()
prevalence = (species_counts / n_cal).to_dict()
# Fallback = plus petite prévalence observée (espèce vue 1 seule fois en calibration)
P_FALLBACK = min(prevalence.values())

# 3c. Score Standard : s = -score(vraie espèce)
# Approche naïve : obs où la vraie espèce est absente du top-k → supprimées
s_std_all = -df_cal_true['score'].values

# 3d. Score PAS : s = -score(vraie espèce) / p̂(y)
cal_p_y = df_cal_true['true_species_id'].map(prevalence).values
s_pas_all = -df_cal_true['score'].values / cal_p_y

# Taille effective de calibration (obs matchées uniquement)
n_cal_eff = len(s_std_all)
log(f"  → Calibration effective (approche naïve) : {n_cal_eff:,} / {n_cal:,} obs")

# ================================================================
# ÉTAPE 4 : ÉVALUATION SUR LE TEST
# ================================================================
log(f"\nConstruction des ensembles de prédiction...")

# Préparer lookup vérité test
test_truth = df_test.set_index('observation_id')['true_species_id'].to_dict()

# Grouper les scores test par observation (une seule fois)
test_groups = {
    obs_id: grp[['species_id', 'score']].values
    for obs_id, grp in df_test_scores.groupby('observation_id')
}

# Vérifier combien d'obs test ont la vraie espèce dans le top-k
n_test_missing = 0
for obs_id in test_id_set:
    if obs_id in test_groups:
        species_in_topk = set(test_groups[o²bs_id][:, 0])
        true_y = test_truth[obs_id]
        if true_y not in species_in_topk:
            n_test_missing += 1
    else:
        n_test_missing += 1
log(f"  Obs test vraie espèce absente du top-k : {n_test_missing} ({100*n_test_missing/n_test:.1f}%)")
log(f"  Plafond structurel de couverture : {100*(1 - n_test_missing/n_test):.1f}%")

results = []

for alpha in ALPHAS:
    # Quantile conforme : ⌈(n+1)(1-α)⌉ / n  avec n = n_cal_eff (obs matchées)
    q_level = np.ceil((n_cal_eff + 1) * (1 - alpha)) / n_cal_eff
    q_level = min(q_level, 1.0)

    q_std = np.quantile(s_std_all, q_level)
    q_pas = np.quantile(s_pas_all, q_level)

    log(f"\n--- α = {alpha:.2f} (cible 1-α = {1-alpha:.2f}) ---")
    log(f"  q_std = {q_std:.6f} | q_pas = {q_pas:.6f}")

    covered_std = []
    covered_pas = []
    size_std    = []
    size_pas    = []
    obs_true_sp = []

    for obs_id in test_id_set:
        true_y = test_truth[obs_id]

        if obs_id not in test_groups:
            # Aucun score IA → ensemble vide → non couvert
            covered_std.append(False)
            covered_pas.append(False)
            size_std.append(0)
            size_pas.append(0)
            obs_true_sp.append(true_y)
            continue

        data = test_groups[obs_id]
        species = data[:, 0]
        scores  = data[:, 1].astype(float)

        # Standard : inclure y si -score(y) ≤ q_std → score(y) ≥ -q_std
        mask_std = scores >= -q_std

        # PAS : inclure y si -score(y)/p̂(y) ≤ q_pas
        # Fallback = plus petite prévalence observée (évite le fallback 1/(2n) trop petit)
        p_y_vec = np.array([prevalence.get(sp, P_FALLBACK) for sp in species])
        mask_pas = (-scores / p_y_vec) <= q_pas

        # Correction top-1 : éviter les ensembles vides (fallback espèce la plus probable)
        if not mask_std.any():
            mask_std[np.argmax(scores)] = True
        if not mask_pas.any():
            mask_pas[np.argmax(scores)] = True

        covered_std.append(true_y in species[mask_std])
        covered_pas.append(true_y in species[mask_pas])
        size_std.append(int(mask_std.sum()))
        size_pas.append(int(mask_pas.sum()))
        obs_true_sp.append(true_y)

    covered_std = np.array(covered_std)
    covered_pas = np.array(covered_pas)
    size_std    = np.array(size_std)
    size_pas    = np.array(size_pas)
    obs_true_sp = np.array(obs_true_sp)

    # --- COUVERTURE MARGINALE ---
    marg_std = covered_std.mean()
    marg_pas = covered_pas.mean()

    # --- COUVERTURE CONDITIONNELLE (par espèce) ---
    unique_sp = np.unique(obs_true_sp)
    cond_std = {}
    cond_pas = {}
    for sp in unique_sp:
        mask = obs_true_sp == sp
        cond_std[sp] = covered_std[mask].mean()
        cond_pas[sp] = covered_pas[mask].mean()

    # --- COUVERTURE MACRO ---
    macro_std = np.mean(list(cond_std.values()))
    macro_pas = np.mean(list(cond_pas.values()))

    # --- TAILLE MOYENNE ET % ENSEMBLES VIDES ---
    avg_size_std = size_std.mean()
    avg_size_pas = size_pas.mean()
    pct_empty_std = 100 * (size_std == 0).sum() / len(size_std)
    pct_empty_pas = 100 * (size_pas == 0).sum() / len(size_pas)

    log(f"  Standard — Marg: {marg_std:.4f} | Macro: {macro_std:.4f} | Taille: {avg_size_std:.2f} | Vides: {pct_empty_std:.1f}%")
    log(f"  PAS      — Marg: {marg_pas:.4f} | Macro: {macro_pas:.4f} | Taille: {avg_size_pas:.2f} | Vides: {pct_empty_pas:.1f}%")
    if marg_std < 1 - alpha and abs(marg_std - (1 - n_test_missing / n_test)) < 0.005:
        log(f"  ⚠ Sous-couverture Standard : plafond structurel ({100*(1-n_test_missing/n_test):.1f}%) < cible ({100*(1-alpha):.0f}%)")

    results.append({
        'alpha': alpha, 'target': 1 - alpha,
        'marg_std': marg_std, 'marg_pas': marg_pas,
        'macro_std': macro_std, 'macro_pas': macro_pas,
        'size_std': avg_size_std, 'size_pas': avg_size_pas,
        'pct_empty_std': pct_empty_std, 'pct_empty_pas': pct_empty_pas,
        'q_std': q_std, 'q_pas': q_pas,
        'cond_std': cond_std, 'cond_pas': cond_pas,
        'size_std_arr': size_std, 'size_pas_arr': size_pas,
    })

# ================================================================
# ÉTAPE 5 : TABLEAU RÉCAPITULATIF
# ================================================================
log(f"\n{'='*80}")
log("TABLEAU RÉCAPITULATIF — CP MARGINALE NAÏVE")
log(f"{'='*80}")
header = f"| {'α':^5} | {'1-α':^5} | {'Marg Std':^9} | {'Marg PAS':^9} | {'Macro Std':^10} | {'Macro PAS':^10} | {'Taille Std':^11} | {'Taille PAS':^11} |"
sep    = f"|{'-'*7}|{'-'*7}|{'-'*11}|{'-'*11}|{'-'*12}|{'-'*12}|{'-'*13}|{'-'*13}|"
log(header)
log(sep)
for r in results:
    log(f"| {r['alpha']:^5.2f} | {r['target']:^5.2f} | {r['marg_std']:^9.4f} | {r['marg_pas']:^9.4f} | {r['macro_std']:^10.4f} | {r['macro_pas']:^10.4f} | {r['size_std']:^11.2f} | {r['size_pas']:^11.2f} |")

# Sauvegarder en CSV
df_results = pd.DataFrame([{
    'alpha': r['alpha'], 'target': r['target'],
    'marg_std': r['marg_std'], 'marg_pas': r['marg_pas'],
    'macro_std': r['macro_std'], 'macro_pas': r['macro_pas'],
    'size_std': r['size_std'], 'size_pas': r['size_pas'],
    'pct_empty_std': r['pct_empty_std'], 'pct_empty_pas': r['pct_empty_pas'],
    'q_std': r['q_std'], 'q_pas': r['q_pas'],
} for r in results])
df_results.to_csv(os.path.join(DIR_SORTIES, "resultats_cp_marginale_naive.csv"), index=False)

# ================================================================
# ÉTAPE 6 : FIGURES (STYLE SANITY CHECK DÉSÉQUILIBRÉ)
# ================================================================
log(f"\nGénération des figures...")

# --- FIGURE 1 : Couverture marginale vs cible (barplot par α) ---
fig, axes = plt.subplots(1, 3, figsize=(21, 6))

x_pos = np.arange(len(ALPHAS))
width = 0.35
targets = [r['target'] for r in results]

axes[0].bar(x_pos - width/2, [r['marg_std'] for r in results], width,
            label='Standard', color='#729ece', edgecolor='black', alpha=0.8)
axes[0].bar(x_pos + width/2, [r['marg_pas'] for r in results], width,
            label='PAS', color='#ff7f0e', edgecolor='black', alpha=0.8)
for i, t in enumerate(targets):
    axes[0].plot([i - 0.45, i + 0.45], [t, t], 'g--', linewidth=1.5,
                 label='Cible' if i == 0 else None)
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels([f"α={a}" for a in ALPHAS])
axes[0].set_ylabel("Couverture")
axes[0].set_title("Couverture Marginale")
axes[0].set_ylim(0, 1.05)
axes[0].legend()

# --- FIGURE 2 : Couverture macro vs cible ---
axes[1].bar(x_pos - width/2, [r['macro_std'] for r in results], width,
            label='Standard', color='#729ece', edgecolor='black', alpha=0.8)
axes[1].bar(x_pos + width/2, [r['macro_pas'] for r in results], width,
            label='PAS', color='#ff7f0e', edgecolor='black', alpha=0.8)
for i, t in enumerate(targets):
    axes[1].plot([i - 0.45, i + 0.45], [t, t], 'g--', linewidth=1.5,
                 label='Cible' if i == 0 else None)
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels([f"α={a}" for a in ALPHAS])
axes[1].set_ylabel("Couverture")
axes[1].set_title("Couverture Macro")
axes[1].set_ylim(0, 1.05)
axes[1].legend()

# --- FIGURE 3 : Taille moyenne des ensembles ---
axes[2].bar(x_pos - width/2, [r['size_std'] for r in results], width,
            label='Standard', color='#729ece', edgecolor='black', alpha=0.8)
axes[2].bar(x_pos + width/2, [r['size_pas'] for r in results], width,
            label='PAS', color='#ff7f0e', edgecolor='black', alpha=0.8)
axes[2].set_xticks(x_pos)
axes[2].set_xticklabels([f"α={a}" for a in ALPHAS])
axes[2].set_ylabel("Taille moyenne")
axes[2].set_title("Taille Moyenne des Ensembles")
axes[2].legend()

fig.suptitle("CP Marginale Naïve — Pl@ntNet (Données Réelles)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(DIR_FIGURES, "fig_cp_marginale_naive_resume.png"), dpi=200, bbox_inches='tight')
log(f"  ✓ fig_cp_marginale_naive_resume.png")

# --- FIGURE 4 : Distribution couverture conditionnelle (α=0.05) ---
# Style sanity check déséquilibré
r05 = [r for r in results if r['alpha'] == 0.05][0]
cond_vals_std = np.array(list(r05['cond_std'].values()))
cond_vals_pas = np.array(list(r05['cond_pas'].values()))

fig2, axes2 = plt.subplots(1, 3, figsize=(21, 6))

# Panel 1 : Histogramme des couvertures conditionnelles
axes2[0].hist(cond_vals_std, bins=20, alpha=0.6, label='Standard', color='#729ece', edgecolor='black')
axes2[0].hist(cond_vals_pas, bins=20, alpha=0.6, label='PAS', color='#ff7f0e', edgecolor='black')
axes2[0].axvline(x=0.95, color='green', linestyle='--', linewidth=2, label='Cible 95%')
axes2[0].set_xlabel("Couverture conditionnelle")
axes2[0].set_ylabel("Nombre d'espèces")
axes2[0].set_title("Distribution des couvertures par espèce (α=0.05)")
axes2[0].legend()

# Panel 2 : 20 espèces les plus fréquentes + 20 les plus rares
sp_counts_test = pd.Series(r05['cond_std']).index
sp_freq = df_test['true_species_id'].value_counts()
top20 = sp_freq.head(20).index.tolist()
bot20 = sp_freq.tail(20).index.tolist()
sel_sp = top20 + bot20

cov_std_sel = [r05['cond_std'].get(sp, 0) for sp in sel_sp]
cov_pas_sel = [r05['cond_pas'].get(sp, 0) for sp in sel_sp]
labels = [f"n={sp_freq[sp]}" for sp in sel_sp]
x_sel = np.arange(len(sel_sp))

axes2[1].bar(x_sel - 0.2, cov_std_sel, 0.4, label='Standard', color='#729ece', alpha=0.8, edgecolor='black')
axes2[1].bar(x_sel + 0.2, cov_pas_sel, 0.4, label='PAS', color='#ff7f0e', alpha=0.8, edgecolor='black')
axes2[1].axhline(y=0.95, color='green', linestyle='--', linewidth=2, label='Cible 95%')
axes2[1].axvline(x=19.5, color='red', linestyle=':', linewidth=1.5, label='Fréq → Rare')
axes2[1].set_xticks(x_sel)
axes2[1].set_xticklabels(labels, rotation=90, fontsize=7)
axes2[1].set_ylabel("Couverture")
axes2[1].set_title("Top 20 fréquentes vs 20 plus rares")
axes2[1].set_ylim(0, 1.1)
axes2[1].legend(fontsize=8)

# Panel 3 : Distribution taille des ensembles
axes2[2].hist(r05['size_std_arr'], bins=range(0, 52), alpha=0.6, label='Standard', color='#729ece', edgecolor='black')
axes2[2].hist(r05['size_pas_arr'], bins=range(0, 52), alpha=0.6, label='PAS', color='#ff7f0e', edgecolor='black')
axes2[2].set_xlabel("Taille de l'ensemble de prédiction")
axes2[2].set_ylabel("Nombre d'observations")
axes2[2].set_title(f"Taille des ensembles (α=0.05)\nStd={r05['size_std']:.1f} | PAS={r05['size_pas']:.1f}")
axes2[2].legend()

fig2.suptitle("Analyse Conditionnelle — CP Naïve α=0.05 (Style Sanity Check)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(DIR_FIGURES, "fig_cp_conditionnelle_naive_alpha005.png"), dpi=200, bbox_inches='tight')
log(f"  ✓ fig_cp_conditionnelle_naive_alpha005.png")

plt.show()

log(f"\n{'='*70}")
log("TERMINÉ")
log(f"{'='*70}")