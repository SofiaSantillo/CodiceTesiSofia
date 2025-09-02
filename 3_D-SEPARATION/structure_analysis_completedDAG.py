import ast
import networkx as nx
import re
from itertools import combinations

def identify_structures(G):
    forks = set()
    chains = set()
    colliders = set()
    
    nodes = list(G.nodes())

    for b in nodes:
        neighbors = list(G.predecessors(b)) + list(G.successors(b))
        
        for a, c in combinations(neighbors, 2):
            # chain: a -> b -> c
            if G.has_edge(a, b) and G.has_edge(b, c):
                chains.add(f"{a} -> {b} -> {c}")
            # chain inversa: c -> b -> a
            if G.has_edge(c, b) and G.has_edge(b, a):
                chains.add(f"{c} -> {b} -> {a}")
            # fork: b -> a, b -> c
            if G.has_edge(b, a) and G.has_edge(b, c):
                forks.add(f"{b} -> {a}, {b} -> {c}")
            # collider: a -> b <- c
            if G.has_edge(a, b) and G.has_edge(c, b):
                colliders.add(f"{a} -> {b} <- {c}")
    
    return list(forks), list(chains), list(colliders)


def main(input_file, output_file):
    with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
        dag_count = 0
        for line in f_in:
            line = line.strip()
            if line.startswith("edges:"):
                # Estrai lista di archi
                match = re.search(r"edges:\s*(\[.*\])", line)
                if not match:
                    f_out.write(f"Errore parsing linea: {line}\n")
                    continue
                try:
                    edges = ast.literal_eval(match.group(1))
                    G = nx.DiGraph(edges)

                    # --- Controllo se il DAG ha cicli ---
                    if not nx.is_directed_acyclic_graph(G):
                        f_out.write(f"DAG #{dag_count} SCARTATO (contiene cicli)\n\n")
                        dag_count += 1
                        continue

                    # Identificazione strutture solo se aciclico
                    forks, chains, colliders = identify_structures(G)

                    print(dag_count)
                    f_out.write(f"DAG #{dag_count}\n")
                    f_out.write(f"Edges: {edges}\n")
                    f_out.write(f"Forks: {forks}\n")
                    f_out.write(f"Chains: {chains}\n")
                    f_out.write(f"Colliders: {colliders}\n")
                    f_out.write("\n")

                    dag_count += 1
                except Exception as e:
                    f_out.write(f"Errore parsing linea: {line} -> {e}\n")

if __name__ == "__main__":
    input_file = "2_DAG/full_DAGs.log"
    output_file = "3_D-SEPARATION/_logs/dag_structures.log"

    main(input_file, output_file)
