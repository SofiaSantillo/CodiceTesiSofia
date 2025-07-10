import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from causaldag import DAG as CausalDAG
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import os
from sklearn.impute import SimpleImputer
from Esploratory_data import run_pipeline
import importlib.util

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


def analyze_dag(dag_str, DAG_name):
    # Se è una stringa tipo "A -> B, B -> C", parse manuale
    if isinstance(dag_str, str):
        print(f"[INFO] Parsing DAG da stringa per {DAG_name}")
        arcs = []
        for relation in dag_str.split(","):
            relation = relation.strip()
            if "->" in relation:
                src, dst = [x.strip() for x in relation.split("->")]
                arcs.append((src, dst))
        dag = CausalDAG(arcs=set(arcs))
    else:
        print(f"[ERRORE] DAG passato non è una stringa.")
        return [], [], [], []

    # Verifica che sia un DAG valido
    if not hasattr(dag, "arcs"):
        print(f"[AVVISO] {DAG_name} non è un oggetto DAG valido dopo il parsing. Skippato.")
        return [], [], [], []

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

                if G.has_edge(A, B) and G.has_edge(C, B):
                    if A < C:
                        colliders.add((A, B, C))
                    else:
                        colliders.add((C, B, A))

                if G.has_edge(B, A) and G.has_edge(B, C):
                    if A < C:
                        forks.add((A, B, C))
                    else:
                        forks.add((C, B, A))

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

    os.makedirs("_Structure_manual_DAG", exist_ok=True)
    structure_file = f"_Structure_manual_DAG/{DAG_name}_structure.txt"

    with open(structure_file, "w", encoding="utf-8") as f:
        f.write(f"{DAG_name} STRUCTURE:\n\n\n")
        f.write("Colliders (A → B ← C):\n")
        f.write("\n".join(format_triplets(colliders, ('→', '←'))) or "None")
        f.write("\n\n")

        f.write("Forks (A ← B → C):\n")
        f.write("\n".join(format_triplets(forks, ('←', '→'))) or "None")
        f.write("\n\n")

        f.write("Chains (A → B → C):\n")
        f.write("\n".join(format_triplets(chains, ('→', '→'))) or "None")
        f.write("\n\n")

        f.write("Backdoor paths (A ⇒ B):\n")
        f.write("\n".join(format_pairs(backdoor_paths)) or "None")
        f.write("\n")

    return list(forks), list(colliders), chains,

def run_pipeline():
    """Main function to execute the analysis of all DAG constructed."""
    for nome, modulo in moduli.items():
        if hasattr(modulo, "constraction_dag"):
            print(f"[INFO] Eseguo {nome}.constraction_dag()")
            dag_str = modulo.constraction_dag() 
            DAG_name = nome.split("Constraction_")[1]
            analyze_dag(dag_str, DAG_name)

if __name__ == "__main__":
    run_pipeline()