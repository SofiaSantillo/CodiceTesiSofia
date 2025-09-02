import json
from itertools import permutations, combinations

def generate_dags(nodes):
    dags = []
    n = len(nodes)
    for perm in permutations(nodes):
        edges_list = []
        for i in range(n):
            for j in range(i+1, n):
                edges_list.append((perm[i], perm[j]))
        for r in range(len(edges_list)+1):
            for subset in combinations(edges_list, r):
                involved_nodes = set()
                for edge in subset:
                    involved_nodes.update(edge)
                if involved_nodes == set(nodes):
                    dags.append(list(subset))
    return dags

with open("2_DAG/_json/generate_all_combination_of_3_nodes.json", "r") as f:
    groups = json.load(f)

with open("2_DAG/_json/vincoli_edges_3n.json", "r") as f:
    forbidden_edges = set(tuple(edge) for edge in json.load(f))

all_dags = {}
num_groups = 116

# Ottieni i gruppi ordinati per ID
group_items = sorted(groups.items(), key=lambda x: int(x[0]))

for group_id_str, nodes in group_items[:num_groups]:
    group_id = int(group_id_str)
    dags = generate_dags(nodes)
    for i, dag in enumerate(dags, start=1):
        if any(tuple(edge) in forbidden_edges for edge in dag):
            continue 
        key = f"{group_id}.{i}"
        all_dags[key] = dag


with open("2_DAG/_json/creation_3nDAG.json", "w") as f:
    json.dump(all_dags, f, indent=4)

print(f"Tutti i DAG generati e salvati. Totale DAG: {len(all_dags)}")

