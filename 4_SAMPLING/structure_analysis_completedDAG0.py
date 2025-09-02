import ast
import networkx as nx
from itertools import combinations

def identify_structures(G):
    forks = set()
    chains = set()
    colliders = set()
    nodes = list(G.nodes())
    for b in nodes:
        neighbors = list(G.predecessors(b)) + list(G.successors(b))
        for a, c in combinations(neighbors, 2):
            if G.has_edge(a, b) and G.has_edge(b, c):
                chains.add(f"{a} -> {b} -> {c}")
            if G.has_edge(c, b) and G.has_edge(b, a):
                chains.add(f"{c} -> {b} -> {a}")
            if G.has_edge(b, a) and G.has_edge(b, c):
                forks.add(f"{b} -> {a}, {b} -> {c}")
            if G.has_edge(a, b) and G.has_edge(c, b):
                colliders.add(f"{a} -> {b} <- {c}")
    return list(forks), list(chains), list(colliders)

def main(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:

        current_dag = ""
        edges_lines = []

        for line in f_in:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#DAG"):
                if edges_lines:
                    process_edges(current_dag, edges_lines, f_out)
                    edges_lines = []
                current_dag = line.replace("#", "")
            else:
                edges_lines.append(line)

        if edges_lines:
            process_edges(current_dag, edges_lines, f_out)

def process_edges(dag_name, edges_lines, f_out):
    edges_str = ''.join(edges_lines) 
    try:
        edges = ast.literal_eval(edges_str)
        G = nx.DiGraph(edges)
        forks, chains, colliders = identify_structures(G)
        f_out.write(f"{dag_name}\n")
        f_out.write(f"Edges: {edges}\n")
        f_out.write(f"Forks: {forks}\n")
        f_out.write(f"Chains: {chains}\n")
        f_out.write(f"Colliders: {colliders}\n\n")
    except Exception as e:
        f_out.write(f"{dag_name} ERRORE: {e}\n\n")

if __name__ == "__main__":
    input_file = "4_SAMPLING/_logs/all_configuration_with_best_dag0.log"
    output_file = "4_SAMPLING/_logs/valid_dags0.log"
    main(input_file, output_file)
