import pandas as pd
from sklearn.model_selection import ShuffleSplit

# 1. Charger le fichier
df = pd.read_csv("observations_experts.csv")

print(f"Total observations : {len(df)}")

# 2. Découpage avec ShuffleSplit (50% calibration, 50% test)
rs = ShuffleSplit(n_splits=1, test_size=0.5, random_state=42)

for train_idx, test_idx in rs.split(df):
    calib_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

# 3. Résultats
print(f"Calibration : {len(calib_df)} observations")
print(f"Test : {len(test_df)} observations")

# 4. Sauvegarder (optionnel)
calib_df.to_csv("calibration_set.csv", index=False)
test_df.to_csv("test_set.csv", index=False)

print("Fichiers sauvegardés : calibration_set.csv et test_set.csv")