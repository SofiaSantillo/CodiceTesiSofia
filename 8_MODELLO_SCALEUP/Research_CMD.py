import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import OrdinalEncoder

# --- Funzione di mutual information condizionata non lineare ---
def conditional_mutual_info(X, Y, Z=None):
    if isinstance(X, pd.Series):
        X = X.values
    if isinstance(Y, pd.Series):
        Y = Y.values
    if Z is not None and isinstance(Z, pd.DataFrame):
        categorical_cols = Z.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            enc = OrdinalEncoder()
            Z[categorical_cols] = enc.fit_transform(Z[categorical_cols])

    if Z is None or (hasattr(Z, 'shape') and Z.shape[1] == 0):
        mi = mutual_info_regression(X.reshape(-1,1), Y)
    else:
        rf_X = RandomForestRegressor(n_estimators=200).fit(Z, X)
        rf_Y = RandomForestRegressor(n_estimators=200).fit(Z, Y)
        res_X = X - rf_X.predict(Z)
        res_Y = Y - rf_Y.predict(Z)
        mi = mutual_info_regression(res_X.reshape(-1,1), res_Y)
    return mi[0]

# --- DFS per verificare se un nodo è sulla catena causale feature -> target ---
def is_on_path(dag, start, end, visited=None):
    if visited is None:
        visited = set()
    if start == end:
        return True
    visited.add(start)
    for child in dag.get(start, []):
        if child not in visited:
            if is_on_path(dag, child, end, visited):
                return True
    return False

# --- Funzione per identificare U, Z e D dal DAG e dal dataset ---
def identify_U_Z_D_from_dag(dag_edges, df, target):
    nodes = df.columns.tolist()
    dag = {node: [] for node in nodes}

    for parent, child in dag_edges:
        dag[parent].append(child)
    
    feature_cols = [n for n in nodes if n != target] #tutti i nodi (nodes) meno il target

    # Step 1: candidati U, Z secondo topologia
    U_candidates, Z_candidates, D_candidates = [], [], []

    for node in nodes:
        if node == target:
            continue
        children = dag.get(node, [])
        parents = [n for n, kids in dag.items() if node in kids]

        # Confondente: genitore comune di feature e target (lasciato come prima)
        if target in children:
            for f in nodes:
                if f != target and f in children:
                    U_candidates.append(node)

        # Mediatore: nodo sulla catena feature -> ... -> target (modifica)
        for feature in feature_cols:
            if is_on_path(dag, feature, target) and node != feature:
                Z_candidates.append(node)

    # Step 2: conferma con dati osservati (non lineare)
    U_confirmed, Z_confirmed, D_confirmed = [], [], []

    df_enc = df.copy()
    cat_cols = df_enc.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        enc = OrdinalEncoder()
        df_enc[cat_cols] = enc.fit_transform(df_enc[cat_cols])

    for u in set(U_candidates):
        mi_XY = conditional_mutual_info(df_enc[feature_cols].values[:,0], df_enc[target].values)
        mi_XY_U = conditional_mutual_info(df_enc[feature_cols].values[:,0], df_enc[target].values, df_enc[[u]])
        if mi_XY_U < mi_XY:
            U_confirmed.append(u)

    for z in set(Z_candidates):
        mi_XY = conditional_mutual_info(df_enc[feature_cols].values[:,0], df_enc[target].values)
        mi_XY_Z = conditional_mutual_info(df_enc[feature_cols].values[:,0], df_enc[target].values, df_enc[[z]])
        if mi_XY_Z < mi_XY:
            Z_confirmed.append(z)

    for node in nodes:
            if node == target:
                continue
            children = dag.get(node, [])
            parents = [n for n, kids in dag.items() if node in kids]

            # Decisione/trattamento: influisce su feature ma non è né U né Z (lasciato come prima)
            for f in nodes:
                if f != target and f in children and node not in U_confirmed + Z_confirmed:
                    D_candidates.append(node)

    for d in set(D_candidates):
        mi_XY = conditional_mutual_info(df_enc[feature_cols].values[:,0], df_enc[target].values)
        mi_XY_D = conditional_mutual_info(df_enc[feature_cols].values[:,0], df_enc[target].values, df_enc[[d]])
        mi_XD = conditional_mutual_info(df_enc[feature_cols].values[:,0], df_enc[d].values)
        if mi_XY_D < mi_XY and mi_XD > 0:
            D_confirmed.append(d)

    return U_confirmed, Z_confirmed, D_confirmed

# --- Log file ---
log_file = "8_MODELLO_SCALEUP/_log/research_CMD.log"
os.makedirs(os.path.dirname(log_file), exist_ok=True)  # crea la cartella se non esiste

def log_print(*args, **kwargs):
    print(*args, **kwargs)
    with open(log_file, "a") as f:
        print(*args, **kwargs, file=f)

# --- Esempio DAG 2 ---
dag_edges = [('FRR', 'AQUEOUS'), ('AQUEOUS', 'PEG'), ('CHOL', 'ESM'), ('ESM', 'HSPC'), ('AQUEOUS', 'TFR'), 
             ('PEG', 'CHOL'), ('PEG', 'PDI'), ('ESM', 'SIZE'), ('ESM', 'PDI'), ('TFR', 'PDI'), ('FRR', 'PDI'), 
             ('TFR', 'SIZE'), ('FRR', 'SIZE')]


df = pd.read_csv("_Data/data_1.csv")

U_SIZE, Z_SIZE, D_SIZE = identify_U_Z_D_from_dag(dag_edges, df, target='SIZE')
U_PDI, Z_PDI, D_PDI = identify_U_Z_D_from_dag(dag_edges, df, target='PDI')

log_print("SIZE:")
log_print("Confondenti:", U_SIZE)
log_print("Mediatori:", Z_SIZE)
log_print("Decisionali:", D_SIZE)

log_print("\nPDI:")
log_print("Confondenti:", U_PDI)
log_print("Mediatori:", Z_PDI)
log_print("Decisionali:", D_PDI)