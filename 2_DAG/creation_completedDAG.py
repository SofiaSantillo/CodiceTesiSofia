import json


INPUT_JSON = "2_DAG/_json/3nDAG_filtered.json"
LOG_PATH = "2_DAG/full_DAGs.log"
max_comb_size = 50
max_results = None

with open(INPUT_JSON, "r") as f:
    data = json.load(f)

# Lista di DAG
all_dags = []
for group_id, entries in data.items():
    for e in entries:
        dag_key = e.get("dag_key")
        edges = e.get("dag_edges", [])
        edges_t = [tuple(edge) for edge in edges]
        nodes = set()
        for a, b in edges_t:
            nodes.add(a)
            nodes.add(b)
        all_dags.append({
            "group": group_id,
            "dag_key": dag_key,
            "edges": edges_t,
            "nodes": nodes
        })

total_3node = len(all_dags)
print(f"Totale DAG da 3 nodi caricati: {total_3node}")


all_nodes_union = set()
for d in all_dags:
    all_nodes_union |= d["nodes"]

print(f"Numero di nodi unici trovati nell'input: {len(all_nodes_union)} -> {sorted(all_nodes_union)}")
TARGET_NODES = all_nodes_union
TARGET_SIZE = len(TARGET_NODES)

# Ordina per utilità
all_dags_sorted = sorted(all_dags, key=lambda x: (-len(x["nodes"]), x["group"], x["dag_key"]))

suffix_union = [set() for _ in range(len(all_dags_sorted)+1)]
for i in range(len(all_dags_sorted)-1, -1, -1):
    suffix_union[i] = suffix_union[i+1] | all_dags_sorted[i]["nodes"]

results = []
seen_keys = set()

def dfs(start_idx, current_indices, current_nodes, current_groups, current_edges_union):
    if current_nodes == TARGET_NODES:
        edges_sorted = tuple(sorted(current_edges_union))
        nodes_sorted = tuple(sorted(current_nodes))
        key = (nodes_sorted, edges_sorted)
        if key not in seen_keys:
            seen_keys.add(key)
            results.append({
                "members": [all_dags_sorted[i]["dag_key"] for i in current_indices],
                "groups": [all_dags_sorted[i]["group"] for i in current_indices],
                "nodes": nodes_sorted,
                "edges": list(edges_sorted)
            })
        return

    if max_results is not None and len(results) >= max_results:
        return
    if len(current_indices) >= max_comb_size:
        return
    if not (current_nodes | suffix_union[start_idx]) >= TARGET_NODES:
        return

    n = len(all_dags_sorted)
    for i in range(start_idx, n):
        entry = all_dags_sorted[i]

        # VINCOLO: al massimo un DAG per gruppo
        if entry["group"] in current_groups:
            continue

        if entry["nodes"].issubset(current_nodes):
            continue

        next_nodes = current_nodes | entry["nodes"]
        next_edges = set(current_edges_union) | set(entry["edges"])
        dfs(i+1,
            current_indices + [i],
            next_nodes,
            current_groups | {entry["group"]},
            next_edges)

        if max_results is not None and len(results) >= max_results:
            return

# ---------- Generazione DAG ----------
dfs(0, [], set(), set(), set())
print(f"Totale combinazioni trovate (copertura completa dei {TARGET_SIZE} nodi): {len(results)}")

# ---------- Filtra DAG senza edge da d-separation ----------
filtered_results = []
for r in results:
    edges_set = set(tuple(e) for e in r["edges"])
    if (('ML', 'PEG') not in edges_set and ('PEG', 'ML') not in edges_set) \
        and (('FRR', 'PEG') not in edges_set and ('PEG', 'FRR') not in edges_set):
        filtered_results.append(r)

print(f"Totale DAG senza edge ML↔PEG nè FRR↔PEG: {len(filtered_results)}")


with open(LOG_PATH, "w", encoding="utf-8") as logf:
    for r in filtered_results:
        logf.write(f"members: {r['members']} | groups: {r['groups']}\n")
        logf.write(f"edges: {r['edges']}\n")
        logf.write("\n" + "-"*60 + "\n\n")

print(f"{len(filtered_results)} DAG completi salvati in log: {LOG_PATH}")


dags_by_nodes = {}
for r in filtered_results:
    dags_by_nodes.setdefault(r["nodes"], []).append(r)

print(f"Numero di gruppi di nodi diversi trovati: {len(dags_by_nodes)}")
