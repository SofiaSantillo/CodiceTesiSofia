import pickle
import re
import sys
import joblib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import shap
import seaborn as sns
from typing import List


# ---------------- RNG per shuffle ----------------
_rng = np.random.default_rng(42)




def calcola_prob_congiunta(dag_edges, df, target_var):
    dag = nx.DiGraph(dag_edges)
    parents = list(dag.predecessors(target_var))


    if parents:
    # Conta congiunta target + genitori
        joint = df.groupby(parents + [target_var]).size().reset_index(name="conteggio")
        # Conta solo genitori
        totals = df.groupby(parents).size().reset_index(name="totale")
        # Merge e calcolo probabilità condizionata
        probs = pd.merge(joint, totals, on=parents)
        probs[f"P_{target_var}"] = probs["conteggio"] / probs["totale"]
        return probs.drop(columns=["conteggio", "totale"])
    else:
    # Nessun genitore → probabilità marginale
        probs = df[target_var].value_counts(normalize=True).reset_index()
        probs.columns = [target_var, f"P_{target_var}"]
        return probs

def propagate_intervention(df, dag_edges, intervention_var, exclude):
    df_new = df.copy()
    dag = nx.DiGraph(dag_edges)
    to_update = [v for v in nx.descendants(dag, intervention_var) if v not in exclude]
    topo_order = list(nx.topological_sort(dag))

    for node in topo_order:
        if node not in to_update:
            continue
        parents = list(dag.predecessors(node))
        if parents:
            group = df_new.groupby(parents + [node]).size().reset_index(name='count')
            totals = df_new.groupby(parents).size().reset_index(name='total')
            group = pd.merge(group, totals, on=parents)
            group[f"P_{node}"] = group['count'] / group['total']
        else:
            counts = df_new[node].value_counts(normalize=True).reset_index()
            counts.columns = [node, f"P_{node}"]
            group = counts

        sampled_values = []
        for idx, row in df_new.iterrows():
            cond = group
            for p in parents:
                cond = cond[cond[p] == row[p]]
            values = cond[node].values
            probs = cond[f"P_{node}"].values
            probs = probs / probs.sum()
            sampled_val = np.random.choice(values, p=probs)
            sampled_values.append(sampled_val)

        df_new[node] = sampled_values
    return df_new

# ---------------- Lettura dati ----------------
df = pd.read_csv("_Data/data_1.csv").dropna()

target_nodes = ["PDI", "SIZE"]
sys.stdout = open(f"6_VALIDAZIONE_INTERVENTISTA/_log/global_analysis.log", "w")

dag_edges = [('FRR', 'AQUEOUS'), ('AQUEOUS', 'PEG'), ('CHOL', 'ESM'), ('ESM', 'HSPC'), 
             ('AQUEOUS', 'TFR'), ('PEG', 'PDI'), ('PEG', 'CHOL'), ('ESM', 'SIZE'), 
             ('ESM', 'PDI'), ('TFR', 'PDI'), ('FRR', 'PDI'), ('TFR', 'SIZE'), ('FRR', 'SIZE')]
dag = nx.DiGraph(dag_edges)

model_path = "_Model/refined_model_size_pdi.pkl"
with open(model_path, "rb") as f:
    model_ = pickle.load(f)

modello_rf = model_["rf_model"]

    
# ---------------- Ciclo interventi ----------------

results = []

for col in df.columns:
    if col in target_nodes:
        continue 
    
    intervention_variable=col
    parents = list(dag.predecessors(intervention_variable))
    dag_do = dag.copy()
    dag_do.remove_edges_from([(p, intervention_variable) for p in parents])

    # Prendi il valore più comune della colonna come intervento
    intervento_valore = df[col].max()
    print(f"\nIntervento su {col}: {intervento_valore}")
    
    df_intervento = df.copy()
    df_intervento[intervention_variable] = intervento_valore
    df_intervento_finale = propagate_intervention(df_intervento, dag_do, intervention_variable, exclude=["SIZE", "PDI"])

    # ---------------- Predizione ML ----------------
    y_pred_pre = modello_rf.predict(df) #predizioni ML pre intervento
    y_pred_post = modello_rf.predict(df_intervento_finale) #predizioni ML post intervento
    
    # ---------------- Predizioni DAG ----------------
    for n in target_nodes:
        print(f"\n{n} -> do ({intervention_variable}= {intervento_valore})")

        prection_dag_pre = calcola_prob_congiunta(dag_edges, df, n) #predizione DAG pre intervento
        expected_dag_post = calcola_prob_congiunta(dag_do, df_intervento_finale, n) #predizione DAG post intervento

    # ---------------- Confronto DAG vs ML ----------------

        # Effetto ML (media variazione)
        delta_ml = (y_pred_post[:, 0] - y_pred_pre[:, 0]) if n == "SIZE" else (y_pred_post[:, 1] - y_pred_pre[:, 1])
        delta_ml_mean = delta_ml.mean()
        print("delta ml medio", delta_ml_mean)

        # Effetto DAG (media variazione)
        delta_dag = expected_dag_post[n] - prection_dag_pre[n]
        delta_dag_mean = delta_dag.mean()
        print("delta dag medio", delta_dag_mean)

        # Coerenza direzionale
        same_direction = (delta_ml_mean * delta_dag_mean) > 0

        print(f"[Confronto effetti su {n}]")
        print(f"Effetto medio ML: {delta_ml_mean:.4f}")
        print(f"Effetto medio DAG: {delta_dag_mean:.16f}")
        print(f"Coerenza direzionale: {same_direction}")


        results.append({
            "intervento": intervention_variable,
            "target": n,
            "ML_effetto_medio": delta_ml_mean,
            "DAG_effetto_medio": delta_dag_mean,
            "direzione_coerente": same_direction
        })
    print("----------------------------------------")

