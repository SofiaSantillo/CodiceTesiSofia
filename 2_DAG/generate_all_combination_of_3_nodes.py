import pandas as pd
from itertools import combinations
import json

df = pd.read_csv("2_DAG/seed_Binn.csv") 
nodes = list(df.columns)

all_combinations = list(combinations(nodes, 3))
combinations_dict = {i+1: list(comb) for i, comb in enumerate(all_combinations)}

with open("2_DAG/_json/generate_all_combination_of_3_nodes.json", "w") as f:
    json.dump(combinations_dict, f, indent=4)

print(f"Numero di combinazioni generate: {len(all_combinations)}")
