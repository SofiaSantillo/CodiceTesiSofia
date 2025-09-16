import ast
import networkx as nx
from itertools import combinations
import re
import re
import ast
from collections import defaultdict

def parse_log_file(input_file):
    """Parsa il file di log e restituisce i DAG come dizionari."""
    dags = {}
    with open(input_file, "r") as f:
        content = f.read()

    blocks = re.split(r"DAG #(\d+)", content)
    for i in range(1, len(blocks), 2):
        dag_id = int(blocks[i])
        block_text = blocks[i+1]

        edges_match = re.search(r"Edges:\s*(\[.*?\])", block_text, re.S)
        colliders_match = re.search(r"Colliders:\s*(\[.*?\])", block_text, re.S)

        edges = ast.literal_eval(edges_match.group(1)) if edges_match else []
        colliders = ast.literal_eval(colliders_match.group(1)) if colliders_match else []

        dags[dag_id] = {
            "edges": edges,
            "colliders": colliders
        }
    return dags


def compute_skeleton(edges):
    """Restituisce lo scheletro come insieme di archi non orientati (frozenset)."""
    skeleton = set()
    for u, v in edges:
        skeleton.add(frozenset((u, v)))
    return frozenset(skeleton)


def filter_colliders(colliders, edges):
    """Ritorna i collider con genitori non adiacenti."""
    edge_set = {frozenset(e) for e in edges}
    valid_colliders = set()

    for c in colliders:
        parts = c.split("->")
        left = parts[0].strip()
        right_part = parts[1].split("<-")
        center = right_part[0].strip()
        right = right_part[1].strip()

        # i genitori sono left e right, non devono essere adiacenti
        if frozenset((left, right)) not in edge_set:
            valid_colliders.add((left, center, right))

    return frozenset(valid_colliders)


def group_equivalence_classes(dags):
    """Raggruppa i DAG in classi di equivalenza."""
    classes = defaultdict(list)
    for dag_id, dag in dags.items():
        skeleton = compute_skeleton(dag["edges"])
        colliders = filter_colliders(dag["colliders"], dag["edges"])
        key = (skeleton, colliders)
        classes[key].append(dag_id)
    return classes


def write_classes_to_file(classes, output_file):
    """Scrive le classi di equivalenza su file di log."""
    with open(output_file, "w") as f:
        for i, (_, dag_ids) in enumerate(classes.items(), 1):
            f.write(f"Classe #{i}: {dag_ids}\n")

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
        content = f_in.read()
        
        dag_blocks = re.split(r"-{10,}", content)
        
        dag_count = 0
        for block in dag_blocks:
            block = block.strip()
            if not block:
                continue

            try:
                dag_match = re.search(r"DAG\s+(\d+):", block)
                if not dag_match:
                    continue
                dag_id = int(dag_match.group(1))

                edges_match = re.search(r"Edges:\s*(\[.*\])", block, re.S)
                if not edges_match:
                    f_out.write(f"Errore: edges non trovati in DAG {dag_id}\n\n")
                    continue
                edges = ast.literal_eval(edges_match.group(1))

                score_match = re.search(r"Score:\s*([0-9\.\-eE]+)", block)
                score = float(score_match.group(1)) if score_match else None

                G = nx.DiGraph(edges)

                if not nx.is_directed_acyclic_graph(G):
                    f_out.write(f"DAG #{dag_id} SCARTATO (contiene cicli)\n\n")
                    dag_count += 1
                    continue

                forks, chains, colliders = identify_structures(G)

                f_out.write(f"DAG #{dag_id}\n")
                f_out.write(f"Edges: {edges}\n")
                f_out.write(f"Score: {score}\n")
                f_out.write(f"Forks: {forks}\n")
                f_out.write(f"Chains: {chains}\n")
                f_out.write(f"Colliders: {colliders}\n")
                f_out.write("\n")

                dag_count += 1

            except Exception as e:
                f_out.write(f"Errore parsing blocco: {e}\n\n")


if __name__ == "__main__":
    input_file = "2_DAG/_Logs/expanded_dags.log"
    output_file = "3_D-SEPARATION/_logs/dag_structures.log"

    main(input_file, output_file)

    input_file = "3_D-SEPARATION/_logs/dag_structures.log"       
    output_file = "3_D-SEPARATION/_logs/equivalence_classes.log"  

    dags = parse_log_file(input_file)
    classes = group_equivalence_classes(dags)
    write_classes_to_file(classes, output_file)

    print(f"Classi di equivalenza salvate in {output_file}")
