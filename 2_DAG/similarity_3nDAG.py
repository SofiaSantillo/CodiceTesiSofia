import json
import numpy as np
import pandas as pd


with open("2_DAG/_json/creation_3nDAG.json", "r") as f:
    all_dags = json.load(f)

group_scores = {}

df = pd.read_csv("2_DAG/seed_Binn.csv")  

group_scores = {}

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
    return similarity

gruppo_scelto = None  

# ---- Filtra i DAG in base al gruppo scelto ----
filtered_dags = {
    k: v for k, v in all_dags.items()
    if gruppo_scelto is None or int(k.split(".")[0]) == gruppo_scelto
}

# ---- Itera sui DAG filtrati e calcola punteggi ----
for dag_key, dag_edges in filtered_dags.items():
    group_id = dag_key.split(".")[0]  
    punteggio = valuta_dag(dag_edges, df)

    if group_id not in group_scores:
        group_scores[group_id] = []
    group_scores[group_id].append({
        "dag_key": dag_key,
        "dag_edges": dag_edges,
        "score": punteggio
    })


# ---- Salva i punteggi ----
with open("2_DAG/_json/similarity_3nDAG.json", "w") as f:
    json.dump(group_scores, f, indent=4)

print("Punteggi calcolati per i gruppi selezionati e salvati.")
num_valid_dags = sum(len(dags) for dags in group_scores.values())
print(f"Numero di DAG validi: {num_valid_dags}")