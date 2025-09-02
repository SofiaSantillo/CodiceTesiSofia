import re
import pandas as pd
import networkx as nx
import numpy as np

# --- 1. Calcolo frequenze marginali solo per radici del DAG ---
def compute_marginals(dag, df):
    marginals = {}
    for node in dag.nodes:
        if dag.in_degree(node) == 0:  # radice
            counts = df[node].value_counts(normalize=True)
            marginals[node] = counts.to_dict()
    return marginals

# --- 2. Calcolo probabilità condizionate solo per nodi con genitori ---
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

# --- 3. Funzione di sampling ---
def sample_from_dict(prob_dict):
    prob_dict = {k: v for k, v in prob_dict.items() if v > 0 and not pd.isna(v)}
    outcomes = list(prob_dict.keys())
    probs = np.array(list(prob_dict.values()), dtype=float)
    probs = probs / probs.sum()
    return np.random.choice(outcomes, p=probs)

# --- 4. Ancestral sampling ---
def ancestral_sampling(dag, marginals, conditionals, n_samples=100):
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
            sample[node] = sample_from_dict(prob_dict)

        if valid:
            samples.append(sample)

    return pd.DataFrame(samples)

# --- 5. Parse DAGs dal nuovo log ---
def parse_dags_from_log(log_file):
    dags = {}
    with open(log_file, "r") as f:
        content = f.read()
    matches = re.findall(r"(DAG\d+)\s+Edges:\s*(\[.*?\])", content, re.DOTALL)
    for dag_name, edges_str in matches:
        edges = eval(edges_str) 
        dag_id = int(re.findall(r'\d+', dag_name)[0])
        dags[dag_id] = edges
    return dags

# --- Main ---
log_file = "4_SAMPLING/_logs/valid_dags0.log"  
dags_dict = parse_dags_from_log(log_file)

data_file = "Data_Droplet/seed_Binning_ordinato.csv" 
df = pd.read_csv(data_file).dropna()

for dag_id, edges in dags_dict.items():
    dag = nx.DiGraph()
    dag.add_edges_from(edges)
    if not nx.is_directed_acyclic_graph(dag):
        print(f"Il DAG {dag_id} contiene un ciclo, impossibile fare ancestral sampling.")
        print("Cicli trovati:", list(nx.simple_cycles(dag)))
    else:
        marginals = compute_marginals(dag, df)
        conditionals = compute_conditionals(dag, df)
        sampled_df = ancestral_sampling(dag, marginals, conditionals, n_samples=270)
        csv_file = f"4_SAMPLING/_csv/_csv_archive/dataset_sampling_DAG{dag_id}.csv"
        sampled_df.to_csv(csv_file, index=False)
