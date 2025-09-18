import ast
import itertools
import os
import random
import re
import pandas as pd
import networkx as nx
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

# --- Parametri ---
log_best_dags = "6_VALIDAZIONE_INTERVENTISTA/_log/DAG2.log"
data_path = "_Data/data_1.csv"
targets = ['SIZE', 'PDI']
out_folder = "6_VALIDAZIONE_INTERVENTISTA/_log"
log_ref_path = "6_VALIDAZIONE_INTERVENTISTA/_log/feature_dag_structures.log"  # aggiorna percorso
df_ref = pd.read_csv(log_ref_path, sep="|")
df_ref.columns = [c.strip() for c in df_ref.columns]  # pulisce spazi
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

import networkx as nx
from itertools import combinations

from typing import List, Sequence

# --- Funzioni per backdoor set ---
# --- Funzioni per backdoor set corrette ---
import os
import pandas as pd
import numpy as np
import networkx as nx
from typing import List, Sequence, Tuple

# --- Funzioni di supporto ---

def pick_minimal_set(sets: Sequence[frozenset]) -> List[str]:
    """
    Restituisce il set più piccolo non vuoto se esiste.
    Se esiste solo il set vuoto, allora restituisce quello.
    """
    non_empty_sets = [s for s in sets if len(s) > 0]
    if non_empty_sets:
        return list(sorted(non_empty_sets[0]))
    elif sets:
        return list(sorted(sets[0]))
    else:
        return []

def all_backdoor_sets_observed(X: str, Y: str, observed_cols: set, DAG: nx.DiGraph) -> List[frozenset]:
    """
    Genera tutti i backdoor sets basati sui genitori di X nel DAG, 
    filtrando solo quelli osservabili.
    """
    parents_X = set(DAG.predecessors(X)) - {X, Y}
    parents_X = {p for p in parents_X if p in observed_cols}
    
    sets = []
    for r in range(len(parents_X)+1):
        for comb in itertools.combinations(parents_X, r):
            sets.append(frozenset(comb))
    return sorted(sets, key=lambda s: (len(s), sorted(s)))

def estimate_effect_adjusted(df: pd.DataFrame, X: str, Y: str, Z: List[str], n_samples=1000, seed=42) -> Tuple[dict, float, str]:
    """
    Stima l'effetto totale causale (TCE) di X su Y usando Monte Carlo con backdoor set Z.

    Args:
        df (pd.DataFrame): dataset.
        X (str): treatment.
        Y (str): outcome.
        Z (List[str]): backdoor set di variabili da aggiustare.
        n_samples (int): numero di campioni Monte Carlo.
        seed (int): seed per riproducibilità.

    Returns:
        Tuple[dict, float, str]: 
            - TCE[x] per ciascun valore di X,
            - effetto totale medio (differenza media tra valori),
            - metodo utilizzato ("montecarlo_tce")
    """
    np.random.seed(seed)
    random.seed(seed)

    df_model = df.copy()
    cat_cols = df_model.select_dtypes(include='object').columns.tolist()
    df_model[cat_cols] = df_model[cat_cols].astype(str)

    x_vals = df_model[X].dropna().unique()
    TCE = {}

    for x in x_vals:
        ey_list = []
        for _ in range(n_samples):
            # Campiona valori di Z dal dataframe
            row = {z: df_model[z].sample(1, replace=True).iloc[0] for z in Z}
            # Imposta intervento su X
            row[X] = x

            # Converti categoriali in stringa
            for col in row:
                if col in cat_cols:
                    row[col] = str(row[col])

            # Filtra righe compatibili
            mask = np.ones(len(df_model), dtype=bool)
            for col, val in row.items():
                mask &= (df_model[col] == val)
            subset = df_model[mask]

            if len(subset) > 0:
                ey_list.append(subset[Y].mean())

        TCE[x] = np.mean(ey_list) if len(ey_list) > 0 else np.nan

    # Calcola effetto totale medio (TCE globale)
    ref = x_vals[0]
    E_ref = TCE[ref]
    diffs = [TCE[x] - E_ref for x in x_vals[1:] if not np.isnan(TCE[x])]
    total_effect = np.mean(diffs) if len(diffs) > 0 else 0.0

    return TCE, total_effect, "montecarlo_tce"


# --- Ciclo principale DAG / Features ---

def audit_for_target(df: pd.DataFrame, G: nx.DiGraph, Y: str) -> pd.DataFrame:
    observed = set(df.columns)
    features = [c for c in df.columns if c != Y]
    rows = []

    for X in features:
        if X not in G.nodes:
            continue
        adjsets = all_backdoor_sets_observed(X, Y, observed, G)
        Z = pick_minimal_set(adjsets)
        tce, eff, method = estimate_effect_adjusted(df, X, Y, Z, n_samples=500, seed=42)
        rows.append({
            "Feature": X,
            "Target": Y,
            "BackdoorSet": Z,
            "Effect": eff,
            "Method": method
        })

    return pd.DataFrame(rows)


# --- Esempio di utilizzo ---
df = pd.read_csv(data_path)
G = nx.DiGraph(edges)  # edges dal tuo DAG
# Prepara log dataframe
df_log = df_ref.copy()
df_log['TCE_Value'] = np.nan
df_log['Percentuale'] = np.nan

for target in ["SIZE", "PDI"]:
    # Audit TCE Monte Carlo
    df_audit = audit_for_target(df, G, target)
    
    for idx, row in df_log.iterrows():
        X = row['Feature'].strip()
        Y = row['Target'].strip()
        
        if X not in df_audit['Feature'].values or Y != target:
            continue
        
        tce = df_audit.loc[df_audit['Feature'] == X, 'Effect'].values[0]
        
        # Percentuale normalizzata
        if Y == "SIZE":
            tce_pct = tce / 10000 * 100
        elif Y == "PDI":
            tce_pct = tce / 1 * 100
        else:
            tce_pct = np.nan
        
        df_log.at[idx, 'TCE_Value'] = tce
        df_log.at[idx, 'Percentuale'] = tce_pct
    
    # Salvataggio log
    log_file_path = os.path.join(out_folder, f"TCE.log")
    with open(log_file_path, "w") as f:
        f.write(f"{'Target':<13}| {'Feature':<17}| {'DAG_Category':<13}| {'TCE_Value':<12}| {'Percentuale':<12}\n")
        for idx, row in df_log.iterrows():
            target = row['Target']
            feature = row['Feature']
            category = row['DAG_Category']
            tce = "" if pd.isna(row['TCE_Value']) else f"{row['TCE_Value']:.6f}"
            pct = "" if pd.isna(row['Percentuale']) else f"{row['Percentuale']:.6f}"
            f.write(f"{target:<12}| {feature:<12}| {category:<12}| {tce:<12}| {pct:<12}\n")

    print(f"TCE DAG per target {target} salvato in: {log_file_path}")
    print(df_audit)
