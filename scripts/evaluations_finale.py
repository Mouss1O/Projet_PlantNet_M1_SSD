"""
Script 4 : Évaluation finale de la Phase 2
"""

import pandas as pd

res = pd.read_csv("resultats.csv")

print("="*50)
print("PHASE 2 - RÉSULTATS FINAUX")
print("="*50)

coverage = res['couvert'].mean()
taille_moyenne = res['taille'].mean()
taille_mediane = res['taille'].median()
ensembles_vides = (res['taille'] == 0).mean()

print(f"\n📊 Métriques globales :")
print(f"   Couverture : {coverage:.3f} (objectif : 0.95)")
print(f"   Taille moyenne : {taille_moyenne:.2f}")
print(f"   Taille médiane : {taille_mediane:.0f}")
print(f"   Ensembles vides : {ensembles_vides:.3f}")

print(f"\n📈 Distribution des tailles :")
dist = res['taille'].value_counts().sort_index()
for taille, count in dist.head(10).items():
    pct = count/len(res)*100
    print(f"   {taille} espèces : {count} obs ({pct:.1f}%)")

# Vérification
if coverage >= 0.95:
    print(f"\n✅ Couverture OK : {coverage:.3f} >= 0.95")
else:
    print(f"\n⚠️ Couverture insuffisante : {coverage:.3f} < 0.95")

# Sauvegarder les métriques
metrics = pd.DataFrame([{
    'phase': '2',
    'coverage': coverage,
    'avg_size': taille_moyenne,
    'median_size': taille_mediane,
    'empty_rate': ensembles_vides
}])
metrics.to_csv("metrics_phase2.csv", index=False)
print("\n✅ Métriques sauvegardées : metrics_phase2.csv")