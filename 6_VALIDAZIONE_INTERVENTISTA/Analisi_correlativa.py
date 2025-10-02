import pickle
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import networkx as nx
import collections
import collections.abc
collections.Iterable = collections.abc.Iterable
from causalgraphicalmodels import CausalGraphicalModel

# -------------------------------
# Processo iterativo di predizione (model_refined)
# -------------------------------

def iterative_refinement_predict(X_raw, rf_model, xgb_size, xgb_pdi, num_epochs=5):
    X = X_raw[rf_model.feature_names_in_].copy()

    initial_preds = rf_model.predict(X)
    size_pred = initial_preds[:, 0].ravel()
    pdi_pred = initial_preds[:, 1].ravel()

    for epoch in range(num_epochs):
        X_with_pdi = X.copy()
        X_with_pdi["PDI"] = pdi_pred
        size_pred = xgb_size.predict(X_with_pdi).ravel()

        X_with_size = X.copy()
        X_with_size["SIZE"] = size_pred
        pdi_pred = xgb_pdi.predict(X_with_size).ravel()

    return pd.DataFrame({
        "Refined_SIZE": size_pred,
        "Refined_PDI": pdi_pred
    })


# ----------------------------------------
# Funzione di sensitivity condizionale
# ----------------------------------------

def conditional_sensitivity(df, target, feature, covariates, rf_model, xgb_size, xgb_pdi, delta=1.0):
    X_full = df.copy()
    X_full = X_full.dropna(subset=[target])
    preds_base = iterative_refinement_predict(X_full, rf_model, xgb_size, xgb_pdi)
    y_base_pred = preds_base[f"Refined_{target}"]
    
    X_inc = X_full.copy()
    
    if pd.api.types.is_numeric_dtype(X_inc[feature]):
        X_inc[feature] = pd.to_numeric(X_inc[feature], errors='coerce')
        if covariates:
            grouped = X_inc.groupby(covariates)
            X_inc[feature] = grouped[feature].transform(lambda x: x + delta)
        else:
            X_inc[feature] += delta
    else:
        categories = X_inc[feature].dropna().unique()
        if len(categories) == 2:
            swap_map = {categories[0]: categories[1], categories[1]: categories[0]}
            X_inc[feature] = X_inc[feature].map(swap_map)
        else:
            X_inc[feature] = X_inc[feature].apply(lambda x: categories[0] if x != categories[0] else categories[1])
    
    preds_inc = iterative_refinement_predict(X_inc, rf_model, xgb_size, xgb_pdi)
    y_inc_pred = preds_inc[f"Refined_{target}"]
    
    sensitivity = np.mean(y_inc_pred - y_base_pred)
    
    return sensitivity


# ----------------------------------------
# Funzione di permutation importance 
# -------------------------------------

def conditional_permutation_importance(df, target, feature, covariates, rf_model, xgb_size, xgb_pdi, n_iter=10, random_state=42):
    """
    Calcola importanza predittiva di 'feature' sul 'target',
    aggiustando per le covariate (backdoor set) usando il processo iterativo.
    """
    rng = np.random.RandomState(random_state)

    X_base = df[[feature] + covariates].copy()
    X_full = df.copy() 
    y = df[target].copy()

    preds = iterative_refinement_predict(X_full, rf_model, xgb_size, xgb_pdi)
    y_pred = preds[f"Refined_{target}"]
    base_score = r2_score(y, y_pred)

    scores = []
    for _ in range(n_iter):
        X_perm = X_full.copy()

        if covariates:
            grouped = df[[feature] + covariates].copy()
            grouped = grouped.groupby(covariates)
            X_perm[feature] = grouped[feature].transform(rng.permutation)
        else:
            X_perm[feature] = rng.permutation(X_perm[feature].values)

        preds_perm = iterative_refinement_predict(X_perm, rf_model, xgb_size, xgb_pdi)
        y_perm_pred = preds_perm[f"Refined_{target}"]
        score = r2_score(y, y_perm_pred)

        scores.append(base_score - score)

    return np.mean(scores)

# ---------------------------------------
# Funzioni per calcolo backdoor path e covariate (stesso procedimento usato in "Analisi_causale.py")
# ---------------------------------------

def all_backdoor_sets_observed(X, Y, observed_cols):
    sets = CGM.get_all_backdoor_adjustment_sets(X, Y) or set()
    filt = [frozenset(s) for s in sets if s.issubset(observed_cols) and X not in s and Y not in s]
    return sorted(set(filt), key=lambda s: (len(s), list(sorted(s))))

def pick_minimal_set(sets):
    return list(sorted(sets[0])) if sets else []


# -------------------------------
# Input
# -------------------------------
df = pd.read_csv("_Data/data_1.csv").dropna()

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
targets = ['SIZE', 'PDI']
features = [f for f in nodes if f not in targets]

model_file = "_Model/refined_model_size_pdi.pkl"
with open(model_file, "rb") as f:
    models = pickle.load(f)

rf_model = models["rf_model"]
xgb_size = models["xgb_size"]
xgb_pdi = models["xgb_pdi"]

# -------------------------------
# Backdoor sets dal DAG
# -------------------------------
CGM = CausalGraphicalModel(nodes=nodes, edges=edges)

observed_cols = set(nodes)
covariates_map = {}
for target in targets:
    covariates_map[target] = {}
    for feature in features:
        sets = all_backdoor_sets_observed(feature, target, observed_cols)
        covariates_map[target][feature] = pick_minimal_set(sets)


# -------------------------------
# Output
# ------------------------------
csv_file = "6_VALIDAZIONE_INTERVENTISTA/_csv/analisi_correlativa.csv"

records = []
log_file = "6_VALIDAZIONE_INTERVENTISTA/_log/analisi_correlativa.log"

with open(log_file, "w") as f:
    f.write(f"{'Target':<8} | {'Feature':<10} | {'Covariates':<15} | {'Permutation Importance (ML)':<10}| {'Sensitivity (ML)':<10}\n")
    f.write("-" * 80 + "\n")
    
    for target in targets:
        for feature in features:
            covariates = covariates_map[target][feature]
            importance = conditional_permutation_importance(df, target, feature, covariates, rf_model, xgb_size, xgb_pdi)
            sensitivity = conditional_sensitivity(df, target, feature, covariates, rf_model, xgb_size, xgb_pdi)

            f.write(f"{target:<8} | {feature:<10} | {str(covariates):<15} | {importance:<27.6f}| {sensitivity:<10.6f}\n")

            records.append({
                "Target": target,
                "Feature": feature,
                "Covariates": ", ".join(covariates),
                "Permutation Importance (ML)": importance,
                "Sensitivity (ML)": sensitivity
            })

df_csv = pd.DataFrame(records)
df_csv.to_csv(csv_file, index=False)

print(f"Analisi completata. Risultati salvati in '{log_file}' e '{csv_file}'.")

print(f"Analisi completata. Risultati salvati in '{log_file}'")
