import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(SCRIPT_DIR, "ai_scores_all.json")

with open(PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

print("Type racine :", type(data))

if isinstance(data, list):
    print("Nombre d'éléments :", len(data))
    print("Premier élément :", data[0])
elif isinstance(data, dict):
    keys = list(data.keys())[:3]
    print("Premières clés :", keys)
    for k in keys:
        print(f"  {k} → {data[k]}")