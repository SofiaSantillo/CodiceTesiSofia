import ast
import os
import re
import pandas as pd
import networkx as nx
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

# --- Parametri ---
log_best_dags = "6_VALIDAZIONE_INTERVENTISTA/_log/DAG2.log"
data_path = "_Data/data_1.csv"
targets = ['SIZE', 'PDI']
out_folder = "6_VALIDAZIONE_INTERVENTISTA/_log"

# --- Funzioni ---
def compute_ACE_general(EY_do):
    """
    Calcola l'ACE generale come media sulle differenze rispetto al valore di riferimento.
    EY_do: dict {x_val: E[Y|do(X=x_val)]}
    """
    x_vals = sorted(EY_do.keys())
    ref = x_vals[0]
    E_ref = EY_do[ref]
    total = 0
    for x_val in x_vals[1:]:
        total += EY_do[x_val] - E_ref
    return total / (len(x_vals)-1) if len(x_vals) > 1 else 0

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

def estimate_EY_do_continuous(df, X, Y, PA, n_points=10, method='rf'):
    """
    Stima E[Y|do(X=x)] usando regressione lineare o Random Forest.
    df: DataFrame con dati
    X: feature di intervento
    Y: target
    PA: lista di genitori/backdoor
    method: 'linear' o 'rf'
    """
    cols = [X] + PA
    df_model = df[cols + [Y]].copy()

    # Identifica colonne categoriche
    cat_cols = df_model.select_dtypes(include='object').columns.tolist()

    if len(cat_cols) > 0:
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        encoded = encoder.fit_transform(df_model[cat_cols])
        encoded_cols = encoder.get_feature_names_out(cat_cols)
        df_encoded = pd.DataFrame(encoded, columns=encoded_cols, index=df_model.index)
        df_model = pd.concat([df_model.drop(columns=cat_cols), df_encoded], axis=1)

    X_model = df_model.drop(columns=[Y])
    y_model = df_model[Y]

    # Scegli modello
    if method == 'linear':
        model = LinearRegression()
    elif method == 'rf':
        model = RandomForestRegressor(n_estimators=200, random_state=42)
    else:
        raise ValueError("method deve essere 'linear' o 'rf'")

    model.fit(X_model, y_model)

    # Valori per do(X)
    if pd.api.types.is_numeric_dtype(df[X]):
        x_vals = np.linspace(df[X].min(), df[X].max(), n_points)
    else:
        x_vals = df[X].dropna().unique()

    EY_do = {}
    for x in x_vals:
        row = X_model.mean().to_frame().T  # valori medi per backdoor
        if pd.api.types.is_numeric_dtype(df[X]):
            row[X] = x
        else:
            for col in X_model.columns:
                if col.startswith(X + "_"):
                    row[col] = 1 if col == f"{X}_{x}" else 0
            if X in row.columns:
                row[X] = x
        EY_do[x] = model.predict(row)[0]
    return EY_do




# --- Legge DAG ---
dag_dict = {}
with open(log_best_dags, "r") as f:
    content = f.read()
pattern = r"DAG (\d+):\s+Edges:\s+(\[.*?\])"
matches = re.findall(pattern, content, re.DOTALL)
for dag_num, edges_str in matches:
    edges = ast.literal_eval(edges_str)
    dag_dict[f"DAG{dag_num}"] = edges

# --- Carica dati ---
df = pd.read_csv(data_path)

# --- Calcolo ACE per tutti i DAG ---
for DAG_name, edges in dag_dict.items():
    G = nx.DiGraph()
    G.add_edges_from(edges)
    log_file_path = os.path.join(out_folder, f"AverageCausalEffect_{DAG_name}.log")
    
    with open(log_file_path, "w") as log_file:
        for X in G.nodes():
            if X in targets:
                continue
            for Y in targets:
                if Y not in G.nodes():
                    continue
                # Backdoor set = genitori di X
                backdoor_set = list(G.predecessors(X))
                
                # Stima E[Y | do(X=x)] usando regressione continua
                EY_do = estimate_EY_do_continuous(df, X, Y, backdoor_set, n_points=10)
                
                # Calcola ACE generale
                ace = compute_ACE_general(EY_do)
                log_file.write(f"ACE generale({X} -> {Y}) = {ace}\n")
    print(f"ACE DAG {DAG_name} salvato in: {log_file_path}")
