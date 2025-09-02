import os
import ast
import networkx as nx
import matplotlib.pyplot as plt

log_folder = "3_D-SEPARATION/_logs"
out_folder= "3_D-SEPARATION"
log_files = ["DAG_IPOTIZZATO.log"]

output_folder = os.path.join(out_folder, "_plots")
os.makedirs(output_folder, exist_ok=True)

for log_file in log_files:
    log_path = os.path.join(log_folder, log_file)

    with open(log_path, "r") as f:
        content = f.read().strip()

    edges = ast.literal_eval(content)

    G = nx.DiGraph()
    G.add_edges_from(edges)

    plt.figure(figsize=(20, 15))
    pos = nx.spring_layout(G, seed=42) 
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue",
            font_size=10, font_weight="bold", arrowsize=30)

    png_name = os.path.splitext(log_file)[0] + ".png"
    png_path = os.path.join(output_folder, png_name)
    plt.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot salvato: {png_path}")
