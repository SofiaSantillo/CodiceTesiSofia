import ast
import re
import os
import re
from collections import defaultdict
import networkx as nx
from matplotlib import pyplot as plt

def find_duplicate_dags(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = re.split(r"\n(?=DAG\s+#\d+)", content.strip())
    dag_dict = {}
    
    for block in blocks:
        id_match = re.search(r"DAG\s+#(\d+)", block)
        if not id_match:
            continue
        dag_id = int(id_match.group(1))

        edges_match = re.search(r"Edges:\s*(\[.*?\])", block, re.DOTALL)
        if not edges_match:
            continue
        edges = eval(edges_match.group(1))
        edge_set = frozenset(tuple(edge) for edge in edges)

        dag_dict[dag_id] = edge_set

    reverse_map = defaultdict(list)
    for dag_id, edge_set in dag_dict.items():
        reverse_map[edge_set].append(dag_id)

    duplicates = {tuple(v): k for k, v in reverse_map.items() if len(v) > 1}

    return duplicates

def parse_and_clean_dags(file_path, output_path):
    dags = {}
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = content.strip().split("----------------------------------------")
    with open(output_path, "w", encoding="utf-8") as out_f:
        for block in blocks:
            if not block.strip():
                continue

            dag_match = re.search(r"DAG\s+(\d+):", block)
            if not dag_match:
                continue
            dag_id = int(dag_match.group(1))

            edges_match = re.search(r"Edges:\s*(\[.*\])", block, re.DOTALL)
            edges = eval(edges_match.group(1)) if edges_match else []

            edges = [e for e in edges if not (
                (e[0] == "AQUEOUS" and e[1] == "PEG") or
                (e[0] == "PEG" and e[1] == "AQUEOUS")
            )]

            

def parse_and_clean_dags(file_path, output_path):
    dags = {}
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = content.strip().split("----------------------------------------")
    with open(output_path, "w", encoding="utf-8") as out_f:
        for block in blocks:
            if not block.strip():
                continue

            dag_match = re.search(r"DAG\s+(\d+):", block)
            if not dag_match:
                continue
            dag_id = int(dag_match.group(1))

            edges_match = re.search(r"Edges:\s*(\[.*\])", block, re.DOTALL)
            edges = eval(edges_match.group(1)) if edges_match else []


            if ("ESM", "PDI") not in edges:
                edges.append(("ESM", "PDI"))

            if ("TFR", "PDI") not in edges:
                edges.append(("TFR", "PDI"))

            if ("FRR", "PDI") not in edges:
                edges.append(("FRR", "PDI"))

            if ("TFR", "SIZE") not in edges:
                edges.append(("TFR", "SIZE"))
            
            if ("FRR", "SIZE") not in edges:
                edges.append(("FRR", "SIZE"))


            score_match = re.search(r"Score:\s*([\d\.eE+-]+)", block)
            score = float(score_match.group(1)) if score_match else None

            dags[dag_id] = {"edges": edges, "score": score}

            out_f.write(f"DAG {dag_id}:\n")
            out_f.write(f"Edges: {edges}\n")
            out_f.write(f"Score: {score}\n")
            out_f.write("-" * 40 + "\n")

    return dags


if __name__ == "__main__":
    input_file = "2_DAG/_Logs/expanded_dags.log"
    output_file = "3_D-SEPARATION/_logs/expanded_Dags_clean.log"
    dags_cleaned = parse_and_clean_dags(input_file, output_file)

    print(f"DAG puliti salvati in {output_file}")

    input_file = "3_D-SEPARATION/_logs/dag_structures.log"  
    duplicates = find_duplicate_dags(input_file)

    if duplicates:
        print("DAG con gli stessi edges:")
        for group in duplicates.keys():
            print(" -", group)
    else:
        print("Nessun DAG duplicato trovato.")

    dag_file = "3_D-SEPARATION/_logs/expanded_Dags_clean.log" 
    output_file = "3_D-SEPARATION/_plot/expanded_Dags_clean..png"

    with open(dag_file, "r") as f:
        content = f.read()

    dag_texts = content.split("Edges:")[1:] 
    dag_edges_list = []

    for dag_text in dag_texts:
        edges_str = dag_text.split("]")[0] + "]"  
        edges = ast.literal_eval(edges_str.strip())
        dag_edges_list.append(edges)

    num_dags = len(dag_edges_list)
    cols = 2
    rows = (num_dags + cols - 1) // cols

    plt.figure(figsize=(15, 7 * rows))

    for i, edges in enumerate(dag_edges_list):
        G = nx.DiGraph()
        G.add_edges_from(edges)

        plt.subplot(rows, cols, i + 1)
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_size=1500, node_color="lightblue",
                font_size=10, font_weight="bold", arrowsize=20)
        plt.title(f"DAG #{i+1}")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Tutti i DAG salvati in un unico plot: {output_file}")



