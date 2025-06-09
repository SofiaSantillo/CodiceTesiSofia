import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from causaldag import DAG
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import os
from sklearn.impute import SimpleImputer
from Esploratory_data import run_pipeline



def analyze_dag(dag):

    G = nx.DiGraph()
    for arc in dag.arcs:
        G.add_edge(*arc)

    forks = set()
    colliders = set()
    chains = []
    backdoor_paths = []

    nodes = list(G.nodes)

    for A in nodes:
        for B in nodes:
            for C in nodes:
                if len({A, B, C}) < 3:
                    continue

                # Collider: A → B ← C (avoid duplicates)
                if G.has_edge(A, B) and G.has_edge(C, B):
                    if A < C:
                        colliders.add((A, B, C))
                    else:
                        colliders.add((C, B, A))

                # Fork: A ← B → C (avoid duplicates)
                if G.has_edge(B, A) and G.has_edge(B, C):
                    if A < C:
                        forks.add((A, B, C))
                    else:
                        forks.add((C, B, A))

                # Chain: A → B → C (all valid triplets)
                if G.has_edge(A, B) and G.has_edge(B, C):
                    chains.append((A, B, C))

    # Backdoor paths
    for A in nodes:
        for B in nodes:
            if A != B and not G.has_edge(A, B):
                if nx.has_path(G, A, B):
                    backdoor_paths.append((A, B))
    
    def format_triplets(triplets, arrows):
        return [f"{t[0]} {arrows[0]} {t[1]} {arrows[1]} {t[2]}" for t in triplets]

    def format_pairs(pairs):
        return [f"{a} ⇒ {b}" for a, b in pairs]

    structure_file = "_Structure_manual_DAG/DAG1.1_structure.txt"
    # Write to file
    with open(structure_file, "w", encoding="utf-8") as f:
        f.write("DAG1.1 STRUCTURE:\n\n\n")
        # Writing collider information
        f.write("Colliders (A → B ← C):\n")
        f.write("\n".join(format_triplets(colliders, ('→', '←'))) or "None")
        f.write("\n\n")
    
        # Writing fork information
        f.write("Forks (A ← B → C):\n")
        f.write("\n".join(format_triplets(forks, ('←', '→'))) or "None")
        f.write("\n\n")
    
        # Writing chain information
        f.write("Chains (A → B → C):\n")
        f.write("\n".join(format_triplets(chains, ('→', '→'))) or "None")
        f.write("\n\n")
    
        # Writing backdoor paths information
        f.write("Backdoor paths (A ⇒ B):\n")
        f.write("\n".join(format_pairs(backdoor_paths)) or "None")
        f.write("\n")

    return list(forks), list(colliders), chains, backdoor_paths


# === DAG Construction ===
df = pd.read_csv("Data_DAG/Nodi_DAG1.csv") 
column_names = df.columns.tolist()
column_names.append("mt")

dag = DAG(set(column_names))
dag.add_arc('SIZE', 'PDI')
dag.add_arc('FRR', 'SIZE')
dag.add_arc('FRR', 'PDI')


# Create networkx graph from dag.arcs
G = nx.DiGraph()
for arc in dag.arcs:
    G.add_edge(*arc)

# Plot and save the DAG
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=5000, font_size=15, font_weight='bold', arrows=True, arrowsize=30)
plt.savefig('_Plot/DAG1.1.png', format='png')
plt.close()

# Analyze structures
forks, colliders, chains, backdoor_paths = analyze_dag(dag)



