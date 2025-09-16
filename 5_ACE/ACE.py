import ast
import sys
import networkx as nx
import itertools
import pandas as pd
import os
import matplotlib.pyplot as plt
import re

# File di log contenente i top DAG
log_best_dags = "5_ACE/_logs/best_DAGs.log"  # sostituire con il percorso corretto

# Legge i DAG migliori dal file di log
dag_dict = {}
with open(log_best_dags, "r") as f:
    content = f.read()

# Pattern per trovare "DAG <numero>:\nEdges: [...]"
pattern = r"DAG (\d+):\s+Edges:\s+(\[.*?\])"
matches = re.findall(pattern, content, re.DOTALL)

for dag_num, edges_str in matches:
    edges = ast.literal_eval(edges_str)
    dag_dict[f"DAG{dag_num}"] = edges

# Carica i dati
data = pd.read_csv('_Data/data_1_Binning.csv')

def compute_ACE_general_reference(P_do):
    """
    Calcola l'ACE generale usando come riferimento il primo valore di X.
    """
    x_vals = sorted(P_do.keys()) 
    ref = x_vals[0]
    E_ref = sum(y * P_do[ref][y] for y in P_do[ref])
    total = 0
    for x_val in x_vals[1:]:
        E_x = sum(y * P_do[x_val][y] for y in P_do[x_val])
        total += E_x - E_ref
    return total / (len(x_vals) - 1) if len(x_vals) > 1 else 0

out_folder = "5_ACE"
log_folder = os.path.join(out_folder, "_logs")
output_folder = os.path.join(out_folder, "_plots")
os.makedirs(output_folder, exist_ok=True)

# Itera su tutti i DAG trovati
for DAG_name, edges in dag_dict.items():
    sys.stdout = open(f"{log_folder}/ACE_{DAG_name}.log", "w")
    
    G = nx.DiGraph()
    G.add_edges_from(edges)

    for Y in G.nodes():
        direct_parents = list(G.predecessors(Y)) 
        if not direct_parents:
            continue
        print(f"\nGenitori diretti di {Y}: {direct_parents}")

        for X in direct_parents:
            print(f"\n--- Calcolo P({Y} | do({X})) e ACE generale ---")

            PA = list(G.predecessors(X))
            P_do = {}

            if not PA:
                for x_val in data[X].unique():
                    P_do[x_val] = {}
                    for y_val in data[Y].unique():
                        prob = len(data[(data[Y]==y_val) & (data[X]==x_val)]) / len(data[data[X]==x_val])
                        P_do[x_val][y_val] = prob
                        print(f"Formula: P({Y}={y_val} | do({X}={x_val})) = P({Y}={y_val} | {X}={x_val})")
                        print(f"Valore: {prob}")
            else:
                PA_values = [data[pa].unique() for pa in PA]
                combinations = list(itertools.product(*PA_values))

                for x_val in data[X].unique():
                    P_do[x_val] = {}
                    for y_val in data[Y].unique():
                        total = 0
                        terms = []
                        for z in combinations:
                            pa_filter = pd.Series([True] * len(data))
                            pa_str = []
                            for idx, pa in enumerate(PA):
                                pa_filter &= (data[pa] == z[idx])
                                pa_str.append(f"{pa}={z[idx]}")

                            pa_count = len(data[pa_filter])
                            if pa_count == 0:
                                continue
                            
                            P_pa = pa_count / len(data)
                            denominator = len(data[(data[X] == x_val) & pa_filter])
                            if denominator == 0:
                                continue
                            numerator = len(data[(data[Y] == y_val) & (data[X] == x_val) & pa_filter])
                            P_y_given_x_pa = numerator / denominator
                            contrib = P_y_given_x_pa * P_pa
                            total += contrib
                            terms.append(f"P({Y}={y_val} | {X}={x_val}, {', '.join(pa_str)}) * P({', '.join(pa_str)})")

                        formula_str = " + ".join(terms)
                        print(f"Formula: P({Y}={y_val} | do({X}={x_val})) = {formula_str}")
                        print(f"Valore: {total}")
                        P_do[x_val][y_val] = total

            ace_ref = compute_ACE_general_reference(P_do)
            print(f"\nACE generale({X} -> {Y}) con riferimento = {ace_ref}")

    # Plot del DAG
    plt.figure(figsize=(20, 15))
    pos = nx.spring_layout(G, seed=2)
    nx.draw(G, pos, with_labels=True, node_size=2500, node_color="lightblue",
            font_size=15, font_weight="bold", arrowsize=25)

    png_name = f"{DAG_name}.png"
    png_path = os.path.join(output_folder, png_name)
    plt.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot salvato: {png_path}")
