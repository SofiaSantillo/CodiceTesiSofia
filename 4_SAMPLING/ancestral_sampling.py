import re
import os
import ast
import pandas as pd
import networkx as nx
import numpy as np

# --- 1. Funzione per il parsing dei DAG dal log ---
def parse_dags_from_log(log_file):
    """
    Restituisce un dizionario {dag_id: edges} leggendo un file log contenente più DAG.
    Il formato atteso è:
    DAG 1:
    Edges: [('A', 'B'), ...]
    Score: 1234.56
    ----------------------------------------
    """
    dags = {}
    with open(log_file, "r", encoding="utf-8") as f:
        content = f.read()

    dag_blocks = re.findall(r"DAG (\d+):\s*Edges: (\[.*?\])\s*Score: .*?\n", content, re.DOTALL)
    
    for dag_id, edges_str in dag_blocks:
        edges = ast.literal_eval(edges_str)
        dags[int(dag_id)] = edges

    return dags

# --- 2. Calcolo frequenze marginali solo per radici del DAG ---
def compute_marginals(dag, df):
    marginals = {}
    for node in dag.nodes:
        if dag.in_degree(node) == 0:  # radice
            counts = df[node].value_counts(normalize=True)
            marginals[node] = counts.to_dict()
    return marginals

# --- 3. Calcolo probabilità condizionate solo per nodi con genitori ---
def compute_conditionals(dag, df):
    conditionals = {}
    for node in dag.nodes:
        parents = list(dag.predecessors(node))
        if not parents:
            continue
        group = df.groupby(parents + [node]).size().reset_index(name='conteggio')
        totali = df.groupby(parents).size().reset_index(name='totale')
        group = pd.merge(group, totali, on=parents)
        group[f"P_{node}"] = group['conteggio'] / group['totale']
        group = group.drop(columns=['conteggio', 'totale'], errors='ignore')
        conditionals[node] = {"parents": parents, "df": group}
    return conditionals

# --- 4. Sampling da dizionario probabilità ---
def sample_from_dict(prob_dict):
    prob_dict = {k: v for k, v in prob_dict.items() if v > 0 and not pd.isna(v)}
    if not prob_dict:
        return None
    outcomes = list(prob_dict.keys())
    probs = np.array(list(prob_dict.values()), dtype=float)
    probs /= probs.sum()
    return np.random.choice(outcomes, p=probs)

# --- 5. Ancestral sampling ---
def ancestral_sampling(dag, marginals, conditionals, n_samples=50, seed=42):
    if seed is not None:
        np.random.seed(seed)

    samples = []
    topo_order = list(nx.topological_sort(dag))

    while len(samples) < n_samples:
        sample = {}
        valid = True
        for node in topo_order:
            parents = list(dag.predecessors(node))
            if not parents:
                prob_dict = marginals[node]
            else:
                cond_df = conditionals[node]["df"]
                mask = np.ones(len(cond_df), dtype=bool)
                for p in parents:
                    mask &= cond_df[p] == sample[p]
                prob_sub = cond_df.loc[mask, [node, f"P_{node}"]]
                if prob_sub.empty:
                    valid = False
                    break
                prob_dict = dict(zip(prob_sub[node], prob_sub[f"P_{node}"]))
            sampled_value = sample_from_dict(prob_dict)
            if sampled_value is None:
                valid = False
                break
            sample[node] = sampled_value
        if valid:
            samples.append(sample)
    return pd.DataFrame(samples)

# --- MAIN SCRIPT ---
log_file = "3_D-SEPARATION/_logs/expanded_Dags_clean.log"
data_file = "_Data/data_1.csv"
output_folder = "4_SAMPLING/_csv"
os.makedirs(output_folder, exist_ok=True)

dags_dict = parse_dags_from_log(log_file)
df = pd.read_csv(data_file).dropna()

for dag_id, edges in dags_dict.items():
    dag = nx.DiGraph()
    dag.add_edges_from(edges)

    if not nx.is_directed_acyclic_graph(dag):
        print(f"[!] Il DAG {dag_id} contiene cicli, impossibile fare ancestral sampling.")
        print("Cicli trovati:", list(nx.simple_cycles(dag)))
        continue

    marginals = compute_marginals(dag, df)
    conditionals = compute_conditionals(dag, df)
    sampled_df = ancestral_sampling(dag, marginals, conditionals, n_samples=50, seed=42)

    csv_file = os.path.join(output_folder, f"dataset_sampling_DAG{dag_id}.csv")
    sampled_df.to_csv(csv_file, index=False)
    print(f"[+] Dataset DAG {dag_id} salvato in: {csv_file}")
