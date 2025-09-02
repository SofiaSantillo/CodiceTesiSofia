import re
import ast
from collections import defaultdict

def parse_log_file(input_file):
    """Parsa il file di log e restituisce i DAG come dizionari."""
    dags = {}
    with open(input_file, "r") as f:
        content = f.read()

    blocks = re.split(r"DAG #(\d+)", content)
    # blocks = ["", "0", "Edges: [...]", "1", "Edges: [...]", ...]
    for i in range(1, len(blocks), 2):
        dag_id = int(blocks[i])
        block_text = blocks[i+1]

        # estrazione edges, colliders ecc.
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


if __name__ == "__main__":
    input_file = "3_D-SEPARATION/_logs/dag_structures.log"       
    output_file = "3_D-SEPARATION/_logs/equivalence_classes.log"  

    dags = parse_log_file(input_file)
    classes = group_equivalence_classes(dags)
    write_classes_to_file(classes, output_file)

    print(f"Classi di equivalenza salvate in {output_file}")
