from itertools import permutations, combinations
import os
import json

def generate_dags(nodes, forbidden_edges):
    dags = []
    seen_dags = set() 
    n = len(nodes)
    for perm in permutations(nodes):
        edges_list = []
        for i in range(n):
            for j in range(i+1, n):
                edges_list.append((perm[i], perm[j]))
        for r in range(2, len(edges_list)+1):
            for subset in combinations(edges_list, r):
                involved_nodes = set()
                for edge in subset:
                    involved_nodes.update(edge)
                if involved_nodes == set(nodes):
                    if all([list(edge) not in forbidden_edges for edge in subset]):
                        dag_key = frozenset(subset)
                        if dag_key not in seen_dags:
                            seen_dags.add(dag_key)
                            dags.append(list(subset))
    return dags

with open("2_DAG/_json/generate_all_combination_of_3_nodes.json", "r") as f:
    groups = json.load(f)

with open("2_DAG/_json/vincoli_edges_3n.json", "r") as f:
    forbidden_edges = json.load(f)

log_folder = "2_DAG/_Logs"
os.makedirs(log_folder, exist_ok=True)
log_file = os.path.join(log_folder, "creation_3nDAG.log")

total_dags = 0 

with open(log_file, "w") as f_log:
    group_items = sorted(groups.items(), key=lambda x: int(x[0]))
    
    for group_id_str, nodes in group_items:
        group_id = int(group_id_str)
        dags = generate_dags(nodes, forbidden_edges)
        for i, dag in enumerate(dags, start=1):
            dag_str = ", ".join([f"{src} -> {dst}" for src, dst in dag])
            f_log.write(f"{group_id}.{i}: {dag_str}\n")
        total_dags += len(dags)

print(f"Tutti i DAG filtrati salvati in log testuale: {log_file}")
print(f"Totale DAG generati e validi (senza archi vincolati e senza duplicati): {total_dags}")
