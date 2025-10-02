import sys
import pandas as pd
import numpy as np
import networkx as nx
from typing import List, Tuple, Sequence
import collections
import collections.abc
collections.Iterable = collections.abc.Iterable
from causalgraphicalmodels import CausalGraphicalModel

# -----------------------------
# 1. Creazione DAG
# -----------------------------
nodes = ['FRR', 'AQUEOUS', 'PEG', 'CHOL', 'ESM', 'HSPC', 'TFR', 'SIZE', 'PDI']
edges = [
    ('FRR', 'AQUEOUS'),
    ('AQUEOUS', 'PEG'),
    ('CHOL', 'ESM'),
    ('ESM', 'HSPC'),
    ('PEG', 'PDI'),
    ('AQUEOUS', 'TFR'),
    ('PEG', 'CHOL'),
    ('ESM', 'SIZE'),
    ('ESM', 'PDI'),
    ('TFR', 'PDI'),
    ('FRR', 'PDI'),
    ('TFR', 'SIZE'),
    ('FRR', 'SIZE')
]


G = nx.DiGraph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)

CGM = CausalGraphicalModel(nodes=nodes, edges=edges)
observed_cols = set(nodes)

targets = ['SIZE', 'PDI']
features = [f for f in nodes if f not in targets]

# -----------------------------
# 2. Classificazione feature DAG
# -----------------------------
def classify_feature(feature, target, dag):
    if dag.has_edge(feature, target):
        return "diretta"
    paths = list(nx.all_simple_paths(nx.DiGraph(dag), source=feature, target=target))
    
    if paths:
        for path in paths:
            is_causal = True
            for i in range(len(path)-1):
                if not dag.has_edge(path[i], path[i+1]):
                    is_causal = False
                    break
            if is_causal:
                return "indiretta"
    
    return "non collegata"

# -----------------------------
# 3. Funzione backdoor sets
# -----------------------------

def all_backdoor_sets_observed(X: str, Y: str, observed_cols: set) -> List[frozenset]:
    sets = CGM.get_all_backdoor_adjustment_sets(X, Y) or set()
    filt = [frozenset(s) for s in sets if s.issubset(observed_cols) and X not in s and Y not in s]
    return sorted(set(filt), key=lambda s: (len(s), list(sorted(s))))

# -----------------------------
# 4. Classificazione feature con backdoor sets
# -----------------------------
def classify_features_with_adjustment(G, targets, observed_cols):
    all_results = {}
    for target in targets:
        results = {}
        nodes_ = [n for n in G.nodes() if n != target]
        for X in nodes_:
            # Percorsi causali
            causal_paths = []
            all_paths = list(nx.all_simple_paths(G.to_undirected(), X, target))
            for path in all_paths:
                is_causal = True
                for i in range(len(path)-1):
                    u, v = path[i], path[i+1]
                    if not G.has_edge(u, v):
                        is_causal = False
                        break
                if is_causal:
                    causal_paths.append(path)
            
            backdoor_sets = all_backdoor_sets_observed(X, target, observed_cols)
            adjust = 'Yes' if backdoor_sets else 'No'
            
            dag_cat = classify_feature(X, target, G)
            
            results[X] = {
                'dag_category': dag_cat,
                'causal_paths': causal_paths,
                'backdoor_sets': backdoor_sets,
                'adjust': adjust
            }
        all_results[target] = results
    return all_results

# -----------------------------
# 5. Funzioni aggiuntive
# -----------------------------
def pick_minimal_set(sets: Sequence[frozenset]) -> List[str]:
    return list(sorted(sets[0])) if sets else []

def estimate_effect_adjusted(df: pd.DataFrame, X: str, Y: str, Z: List[str]) -> Tuple[float, str]:
    print(X)
    print(Y)
    print(Z)
    cols = [c for c in [Y, X] + Z if c in df.columns]
    print(cols)
    data = df[cols].dropna()
    if len(data) < 30:
        return (np.nan, "insufficient_data")

    y = data[Y]
    XZ = pd.get_dummies(data[[X] + Z], drop_first=True)
    print(XZ)
    x_cols = [X] if X in XZ.columns else [c for c in XZ.columns if c.startswith(f"{X}_")]
    if not x_cols:
        return (np.nan, "x_not_in_design_matrix")

    try:
        import statsmodels.api as sm
        XZ_sm = sm.add_constant(XZ, has_constant="add")
        if pd.api.types.is_numeric_dtype(y):
            model_sm = sm.OLS(y, XZ_sm).fit()
            eff = float(model_sm.params.get(x_cols[0], np.nan)) if len(x_cols) == 1 else float(np.sum(np.abs([model_sm.params.get(c, 0.0) for c in x_cols])))
            return (eff, "statsmodels_ols")
        else:
            y_bin = y.astype("category").cat.codes
            model_sm = sm.Logit(y_bin, XZ_sm).fit(disp=False)
            eff = float(model_sm.params.get(x_cols[0], np.nan)) if len(x_cols) == 1 else float(np.sum(np.abs([model_sm.params.get(c, 0.0) for c in x_cols])))
            return (eff, "statsmodels_logit")
    except Exception:
        from sklearn.linear_model import LinearRegression, LogisticRegression
        if pd.api.types.is_numeric_dtype(y):
            lr = LinearRegression().fit(XZ, y)
            coef_map = dict(zip(XZ.columns, lr.coef_))
            eff = float(coef_map.get(x_cols[0], np.nan)) if len(x_cols) == 1 else float(np.sum([coef_map.get(c, 0.0) for c in x_cols]))
            return (eff, "sklearn_ols")
        else:
            y_bin = y.astype("category").cat.codes
            clf = LogisticRegression(max_iter=1000).fit(XZ, y_bin)
            coef_map = dict(zip(XZ.columns, clf.coef_.ravel()))
            eff = float(coef_map.get(x_cols[0], np.nan)) if len(x_cols) == 1 else float(np.sum([coef_map.get(c, 0.0) for c in x_cols]))
            return (eff, "sklearn_logit")

# -----------------------------
# 6. Caricamento dataset e analisi
# -----------------------------
df = pd.read_csv("_Data/data_1.csv").dropna()

results = classify_features_with_adjustment(G, targets, observed_cols)


# -----------------------------
# 7. Salvataggio log integrato con effetto causale
# -----------------------------
from dowhy import CausalModel
records = []

log_file_path = "6_VALIDAZIONE_INTERVENTISTA/_log/analisi_causale.log"
csv_file_path = "6_VALIDAZIONE_INTERVENTISTA/_csv/analisi_causale.csv"
with open(log_file_path, "w") as f:
    f.write(f"{'Target':<10} |{'Feature':<10} | {'DAG_Category':<13} | {'Covariates':<21} | {'Adj?':<5} | {'Est_Effect':<22} | {'Method':<15} | {'Causal_Effect':<15}\n")
    f.write("-"*150 + "\n")
    for target, features in results.items():
        
        for feature, info in features.items():
            min_set = pick_minimal_set(info['backdoor_sets'])
            adjust = 'Yes' if min_set else 'No'
            
            est_eff, method = estimate_effect_adjusted(df, feature, target, min_set)
        
            # -----------------------------
            # Calcolo effetto causale reale con DoWhy
            # -----------------------------
            causal_eff = np.nan
            if min_set:  
                try:
                    model_dowhy = CausalModel(
                        data=df,
                        treatment=feature,
                        outcome=target,
                        common_causes=min_set
                    )
                    identified_estimand = model_dowhy.identify_effect()
                    causal_estimate = model_dowhy.estimate_effect(
                        identified_estimand,
                        method_name="backdoor.linear_regression"
                    )
                    causal_eff = causal_estimate.value
                except Exception as e:
                    causal_eff = f"error: {str(e)}"

            f.write(f"{target:<10} |{feature:<10} | {info['dag_category']:<13} | {str(min_set):<21} | {adjust:<5} | {est_eff:<22} | {method:<15} | {causal_eff:<15}\n")
            records.append({
                "Target": target,
                "Feature": feature,
                "DAG_Category": info['dag_category'],
                "Minimal Covariates": str(min_set),
                "Adj?": adjust,
                "Est_Effect": est_eff,
                "Method": method,
                "Causal_Effect": causal_eff
            })

df_csv = pd.DataFrame(records)
df_csv.to_csv(csv_file_path, index=False)

print(f"Analisi completata. Risultati salvati in '{log_file_path}' e '{csv_file_path}'.")

