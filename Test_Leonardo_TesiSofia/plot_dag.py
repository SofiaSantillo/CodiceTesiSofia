import os
import pickle
import networkx as nx
import matplotlib.pyplot as plt

pkl_path = "Test_Leonardo_TesiSofia/_logs/dag_nx.pkl"   
out_folder = "Test_Leonardo_TesiSofia/_plots"
os.makedirs(out_folder, exist_ok=True)

# scegli i DAG che vuoi plottare
dags_to_plot = ["DAG1_IPOTIZZATO", "DAG2", "DAG3", "DAG4", "DAG5", "DAG6", "DAG7", "DAG8", "DAG9", "DAG10", "DAG11"]

with open(pkl_path, "rb") as f:
    graphs = pickle.load(f)

for dag_name in dags_to_plot:
    if dag_name not in graphs:
        print(f" {dag_name} non trovato nel pickle. Disponibili: {list(graphs.keys())}")
        continue

    G = graphs[dag_name]

    plt.figure(figsize=(20, 15))
    pos = nx.spring_layout(G, seed=2)  
    nx.draw(
        G, pos, with_labels=True,
        node_size=2000, node_color="lightblue",
        font_size=10, font_weight="bold", arrowsize=20
    )

    png_name = f"{dag_name}.png"
    png_path = os.path.join(out_folder, png_name)
    plt.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot salvato: {png_path}")

