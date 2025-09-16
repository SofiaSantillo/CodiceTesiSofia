import os
import pandas as pd
import numpy as np
import itertools
from math import log

def read_dags_from_log(log_file_path):
    all_dags = {}
    with open(log_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            dag_id, edges_str = line.split(":")
            edges = []
            for edge in edges_str.split(","):
                edge = edge.strip()
                if "->" in edge:
                    src, tgt = edge.split("->")
                    edges.append((src.strip(), tgt.strip()))
            all_dags[dag_id.strip()] = edges
    return all_dags

log_file_path = "2_DAG/_Logs/creation_3nDAG.log"  
all_dags = read_dags_from_log(log_file_path)

df = pd.read_csv("_Data/data_1_Binning.csv")

def calcola_probabilita_empirica(df, colonne):
    totale = len(df)
    probabilita = df.groupby(colonne).size().reset_index(name='conteggio')
    probabilita['P_empirica'] = probabilita['conteggio'] / totale
    probabilita = probabilita.drop(columns='conteggio')
    return probabilita

def calcola_prob_dag(dag, selected_columns, df):
    prob_df = pd.DataFrame()
    for col in selected_columns:
        parents = [a for (a, b) in dag if b == col]
        group_cols = parents + [col] if parents else [col]
        group = df.groupby(group_cols).size().reset_index(name='conteggio')
        if parents:
            totali = df.groupby(parents).size().reset_index(name='totale')
            group = pd.merge(group, totali, on=parents)
            group[f"P_{col}"] = group['conteggio'] / group['totale']
            group = group.drop(columns=['conteggio', 'totale'], errors='ignore')
        else:
            group[f"P_{col}"] = group['conteggio'] / len(df)
            group = group.drop(columns='conteggio')
        if prob_df.empty:
            prob_df = group
        else:
            common_cols = list(set(prob_df.columns) & set(group.columns))
            if common_cols:
                prob_df = pd.merge(prob_df, group, on=common_cols, how='outer')
            else:
                prob_df = pd.concat([prob_df, group], axis=1)
    prob_cols = [col for col in prob_df.columns if col.startswith('P_')]
    prob_df['P_fattorizzata'] = prob_df[prob_cols].prod(axis=1)
    return prob_df

def similarity_score(empirica, fattorizzata):
    empirica = np.array(empirica)
    fattorizzata = np.array(fattorizzata)
    epsilon = 1e-10
    somma = empirica + fattorizzata + epsilon
    diff = np.abs(empirica - fattorizzata)
    similarity = 1 - (diff / somma)
    return np.mean(similarity)

def confronta_empirica_vs_fattorizzata(df, selected_columns, fattorizzata_df):
    empirica_df = calcola_probabilita_empirica(df, selected_columns)
    merged = pd.merge(empirica_df, fattorizzata_df, on=selected_columns, how='outer')
    merged['P_empirica'] = merged['P_empirica'].fillna(0)
    merged['P_fattorizzata'] = merged['P_fattorizzata'].fillna(0)
    return similarity_score(merged['P_empirica'], merged['P_fattorizzata'])

def valuta_dag(dag, df):
    dag_nodes = set([n for edge in dag for n in edge])
    selected_columns = [col for col in df.columns if col in dag_nodes]
    fattorizzata_df = calcola_prob_dag(dag, selected_columns, df)
    similarity = confronta_empirica_vs_fattorizzata(df[selected_columns], selected_columns, fattorizzata_df)
    return similarity, fattorizzata_df

def get_cpt(data, child, parents):
    if not parents:
        counts = data[child].value_counts(normalize=True)
        return {None: counts.to_dict()}
    counts = (
        data.groupby(parents + [child], observed=True)
        .size()
        .reset_index(name="count")
    )
    parent_counts = (
        data.groupby(parents, observed=True)
        .size()
        .reset_index(name="total")
    )
    merged = pd.merge(counts, parent_counts, on=parents)
    merged["prob"] = merged["count"] / merged["total"]
    cpt = {}
    for _, row in merged.iterrows():
        key = tuple(row[p] for p in parents)
        if key not in cpt:
            cpt[key] = {}
        cpt[key][row[child]] = row["prob"]
    return cpt
     

def log_likelihood(data, dag):
    cpts = {}
    nodes = set(itertools.chain(*dag))
    for node in nodes:
        parents = [u for (u, v) in dag if v == node]
        cpts[node] = get_cpt(data, node, parents)
    ll = 0.0
    for _, row in data.iterrows():
        p = 1.0
        for node in nodes:
            parents = [u for (u, v) in dag if v == node]
            # marginal probability
            if not parents:
                prob = cpts[node][None].get(row[node], 1e-6)
            # conditional probability
            else:
                key = tuple(row[p] for p in parents)
                prob = cpts[node].get(key, {}).get(row[node], 1e-6)
            p *= prob
        # add log probability (with small epsilon for stability)
        ll += log(p + 1e-12)
    return ll


def num_params(data, dag):
    params = 0
    nodes = set(itertools.chain(*dag))
    for node in nodes:
        parents = [u for (u, v) in dag if v == node]
        states_child = data[node].nunique()
        if not parents:
            # If no parents: need (states - 1) parameters for multinomial distribution
            params += states_child - 1
        else:
            # If parents exist: parameters grow with product of parent state counts
            parent_states = np.prod([data[p].nunique() for p in parents])
            params += parent_states * (states_child - 1)
    return params

def score_dag(data, dag):
    ll = log_likelihood(data, dag)
    k = num_params(data, dag)
    n = len(data)
    aic = 2 * k - 2 * ll
    bic = log(n) * k - 2 * ll
    return ll, aic, bic

all_dags = read_dags_from_log(log_file_path)

output_log_path = "2_DAG/_Logs/3nDAG_with_scores.log"
os.makedirs(os.path.dirname(output_log_path), exist_ok=True)

num_valid_dags = 0

with open(output_log_path, "w") as log_file:
    for dag_key, dag_edges in all_dags.items():
        sim_score, _ = valuta_dag(dag_edges, df)
        ll, aic, bic = score_dag(df, dag_edges)
        edges_str = ", ".join([f"{src} -> {tgt}" for src, tgt in dag_edges])
        log_file.write(f"{dag_key}: {edges_str} | similarity_score: {sim_score:.4f} | LL: {ll:.2f} | AIC: {aic:.2f} | BIC: {bic:.2f}\n")
        num_valid_dags += 1

print(f"Tutti i punteggi calcolati. Numero di DAG validi: {num_valid_dags}")
print(f"Risultati salvati in {output_log_path}")