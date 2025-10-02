import re
import networkx as nx
import matplotlib.pyplot as plt

log_file = "5_ACE/_logs/best_dags.log"   
output_file = "5_ACE/top3_sampling_dags.png"

def parse_dags_from_log(filepath):
    dags = {}
    with open(filepath, "r") as f:
        text = f.read()

    pattern = r"DAG (\d+):\s*Edges: \[(.*?)\]"
    matches = re.findall(pattern, text, re.S)

    for dag_num, edges_str in matches:
        edges = re.findall(r"\('([^']+)', '([^']+)'\)", edges_str)
        dags[int(dag_num)] = edges

    return dags

# === PLOTTING ===
def plot_dags(dags, output_file):
    n_dags = len(dags)
    fig, axes = plt.subplots(1, n_dags, figsize=(7 * n_dags, 7))

    if n_dags == 1:
        axes = [axes]

    for ax, (dag_num, edges) in zip(axes, dags.items()):
        G = nx.DiGraph()
        G.add_edges_from(edges)

        pos = nx.spring_layout(G, seed=42, k=1.5, iterations=200)

        nx.draw(
            G,
            pos,
            with_labels=True,
            node_size=2000,
            node_color="lightblue",
            edge_color="gray",
            arrows=True,
            arrowsize=20,
            font_size=8,
            font_weight="bold",
            ax=ax
        )
        ax.set_title(f"DAG {dag_num}", fontsize=16, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"[INFO] Plot salvato in: {output_file}")

if __name__ == "__main__":
    dags = parse_dags_from_log(log_file)
    if not dags:
        print("[ERRORE] Nessun DAG trovato nel log!")
    else:
        plot_dags(dags, output_file)