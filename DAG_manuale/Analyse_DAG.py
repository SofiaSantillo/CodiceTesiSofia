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
import importlib.util
import os

moduli = {}
cartella = "DAG_manuale"

for file in os.listdir(cartella):
    if file.startswith("Constraction_DAG") and file.endswith(".py"):
        nome_modulo = file[:-3]  # Rimuove .py
        percorso = os.path.join(cartella, file)

        spec = importlib.util.spec_from_file_location(nome_modulo, percorso)
        modulo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(modulo)

        moduli[nome_modulo] = modulo


def analyze_dag(dag, DAG):
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

    structure_file = f"_Structure_manual_DAG/{DAG}_structure.txt"
    # Write to file
    with open(structure_file, "w", encoding="utf-8") as f:
        f.write(f"{DAG} STRUCTURE:\n\n\n")
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


def run_pipeline():
    """Main function to execute the analysis of all DAG constructed."""
    # Itera su ogni file CSV nella cartella
    for nome, modulo in moduli.items():
        if hasattr(modulo, "constraction_dag"):
            print(f"Eseguo {nome}.constraction_dag()")
            dag=modulo.constraction_dag()
            DAG = nome.split("Constraction_")[1]
            analyze_dag(dag=dag, DAG=DAG)

            
if __name__ == "__main__":
    run_pipeline()