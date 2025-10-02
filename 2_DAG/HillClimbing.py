import itertools
import networkx as nx
import numpy as np
import pandas as pd
from math import log
import re
import matplotlib.pyplot as plt

# ------------------------------
# Funzioni CPT, log-likelihood e score
# ------------------------------
def get_cpt(data, child, parents, alpha=1.0):
    child_states = data[child].unique()
    cpt = {}
    if not parents:
        counts = data[child].value_counts()
        total = counts.sum()
        for val in child_states:
            cpt.setdefault(None, {})[val] = (counts.get(val, 0) + alpha) / (total + alpha * len(child_states))
        return cpt
    counts = data.groupby(parents + [child]).size().reset_index(name="count")
    parent_counts = data.groupby(parents).size().reset_index(name="total")
    merged = pd.merge(counts, parent_counts, on=parents)
    for _, row in merged.iterrows():
        key = tuple(row[p] for p in parents)
        if key not in cpt:
            cpt[key] = {}
        cpt[key][row[child]] = (row["count"] + alpha) / (row["total"] + alpha * len(child_states))
    for _, row in parent_counts.iterrows():
        key = tuple(row[p] for p in parents)
        if key not in cpt:
            cpt[key] = {}
        total = row["total"]
        for val in child_states:
            if val not in cpt[key]:
                cpt[key][val] = alpha / (total + alpha * len(child_states))
    return cpt

def log_likelihood(data, dag, alpha=1.0):
    variables = list(data.columns)
    cpts = {}
    for node in variables:
        parents = [u for (u, v) in dag if v == node]
        cpts[node] = get_cpt(data, node, parents, alpha=alpha)
    ll = 0.0
    for _, row in data.iterrows():
        for node in variables:
            parents = [u for (u, v) in dag if v == node]
            if not parents:
                prob = cpts[node][None].get(row[node], 1e-12)
            else:
                key = tuple(row[p] for p in parents)
                prob = cpts[node].get(key, {}).get(row[node], 1e-12)
            ll += log(prob)
    return ll

def num_params(data, dag):
    variables = list(data.columns)
    params = 0
    for node in variables:
        parents = [u for (u, v) in dag if v == node]
        states_child = data[node].nunique()
        if not parents:
            params += states_child - 1
        else:
            parent_states = np.prod([data[p].nunique() for p in parents])
            params += parent_states * (states_child - 1)
    return params

def score_dag(data, dag, alpha=1.0):
    ll = log_likelihood(data, dag, alpha=alpha)
    k = num_params(data, dag)
    n = len(data)
    aic = 2 * k - 2 * ll
    bic = log(n) * k - 2 * ll
    return ll, aic, bic

# ------------------------------
# Funzioni log e vincoli
# ------------------------------
def load_allowed_edges(log_path):
    allowed_edges = set()
    with open(log_path, "r") as f:
        for line in f:
            edges = re.findall(r"(\w+)\s*->\s*(\w+)", line)
            allowed_edges.update([tuple(edge) for edge in edges])
    return allowed_edges

def violates_constraints(candidate_edges, allowed_edges):
    return any(edge not in allowed_edges for edge in candidate_edges)

# ------------------------------
# Hill Climbing con vincoli
# ------------------------------
def hill_climb_with_constraints(data, score_metric="BIC", max_iter=1000, start_dag=None,
                                candidate_edges=None, alpha=1.0, allowed_edges=None):
    
    variables = list(data.columns)
    dag = start_dag if start_dag else []
    ll, aic, bic = score_dag(data, dag, alpha=alpha)
    score = {"LL": ll, "AIC": aic, "BIC": bic}[score_metric]
    steps = 0
    improved = True

    while improved and steps < max_iter:
        improved = False
        best_move = None
        best_score = score
        candidates = candidate_edges if candidate_edges is not None else itertools.permutations(variables, 2)
        for u, v in candidates:
            if u == v:
                continue

            # --- Try add ---
            if (u, v) not in dag:
                new_dag = dag + [(u, v)]
                G = nx.DiGraph(new_dag)
                if nx.is_directed_acyclic_graph(G) and not violates_constraints(new_dag, allowed_edges):
                    ll_new, aic_new, bic_new = score_dag(data, new_dag, alpha=alpha)
                    new_score = {"LL": ll_new, "AIC": aic_new, "BIC": bic_new}[score_metric]
                    if (score_metric == "LL" and new_score > best_score) or \
                       (score_metric in ["AIC", "BIC"] and new_score < best_score):
                        best_move = ("add", (u, v))
                        best_score = new_score

            # --- Try remove ---
            if (u, v) in dag:
                new_dag = [e for e in dag if e != (u, v)]
                G = nx.DiGraph(new_dag)
                if nx.is_directed_acyclic_graph(G) and not violates_constraints(new_dag, allowed_edges):
                    ll_new, aic_new, bic_new = score_dag(data, new_dag, alpha=alpha)
                    new_score = {"LL": ll_new, "AIC": aic_new, "BIC": bic_new}[score_metric]
                    if (score_metric == "LL" and new_score > best_score) or \
                       (score_metric in ["AIC", "BIC"] and new_score < best_score):
                        best_move = ("remove", (u, v))
                        best_score = new_score

            # --- Try invert ---
            if (u, v) in dag and (v, u) not in dag:
                new_dag = [e for e in dag if e != (u, v)] + [(v, u)]
                G = nx.DiGraph(new_dag)
                if nx.is_directed_acyclic_graph(G) and not violates_constraints(new_dag, allowed_edges):
                    ll_new, aic_new, bic_new = score_dag(data, new_dag, alpha=alpha)
                    new_score = {"LL": ll_new, "AIC": aic_new, "BIC": bic_new}[score_metric]
                    if (score_metric == "LL" and new_score > best_score) or \
                       (score_metric in ["AIC", "BIC"] and new_score < best_score):
                        best_move = ("invert", (u, v))
                        best_score = new_score 

        # --- Apply best move ---
        if best_move:
            action, edge = best_move
            if action == "add":
                dag.append(edge)
            elif action == "remove":
                dag.remove(edge)
            elif action == "invert":
                dag.remove(edge)
                dag.append((edge[1], edge[0]))
            score = best_score
            improved = True
            steps += 1

    return dag, score

# ------------------------------
# Espansione DAG dal log
# ------------------------------
def parse_log(log_path):
    dags = []
    with open(log_path, "r") as f:
        for line in f:
            edges = re.findall(r"(\w+)\s*->\s*(\w+)", line)
            dags.append(edges)
    return dags

def expand_dag_with_hill_climb_constraints(data, base_dag_edges, allowed_edges, score_metric="BIC", max_iter=1000, alpha=1.0):
    start_dag = [tuple(edge) for edge in base_dag_edges]
    variables = list(data.columns)
    candidate_edges = [e for e in itertools.permutations(variables, 2) if e not in start_dag]
    final_dag, final_score = hill_climb_with_constraints(
        data, score_metric, max_iter, start_dag, candidate_edges, alpha, allowed_edges
    )
    return final_dag, final_score

def expand_dags_from_log_constraints(data, log_path, allowed_edges_file, score_metric="BIC", max_iter=1000, alpha=1.0):
    base_dags = parse_log(log_path)
    allowed_edges = load_allowed_edges(allowed_edges_file)
    expanded_dags = []
    for base_dag in base_dags:
        final_dag, final_score = expand_dag_with_hill_climb_constraints(
            data, base_dag, allowed_edges, score_metric, max_iter, alpha
        )
        expanded_dags.append({"dag": final_dag, "score": final_score})
    return expanded_dags

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    dataset = pd.read_csv("_Data/data_1_Binning.csv")
    log_file = "2_DAG/_Logs/top_3nDAG.log"
    allowed_edges_file = "2_DAG/_Logs/3nDAG_valid_with_scores.log"
    output_log_file = "2_DAG/_Logs/expanded_dags.log"
    output_png_file = "2_DAG/_Plot/expanded_dags.png"

    expanded_dags = expand_dags_from_log_constraints(
        dataset, log_file, allowed_edges_file, score_metric="BIC", max_iter=1000
    )

    with open(output_log_file, "w") as f:
        for i, d in enumerate(expanded_dags, 1):
            f.write(f"DAG {i}:\n")
            f.write(f"Edges: {d['dag']}\n")
            f.write(f"Score: {d['score']}\n")
            f.write("-" * 40 + "\n")

    n_dags = len(expanded_dags)
    n_cols = 2
    n_rows = (n_dags + 1) // n_cols
    plt.figure(figsize=(n_cols * 6, n_rows * 5))

    for idx, d in enumerate(expanded_dags, 1):
        plt.subplot(n_rows, n_cols, idx)
        G = nx.DiGraph(d["dag"])
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=2000, arrowsize=20)
        bic_score = d["score"] if isinstance(d["score"], (int, float)) else d["score"].get("BIC", "")
        plt.title(f"DAG {idx} | BIC: {bic_score:.2f}")

    plt.tight_layout()
    plt.savefig(output_png_file)
    plt.close()
