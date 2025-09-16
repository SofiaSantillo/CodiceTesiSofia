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


# ---------------- Funzioni Sensitivity / Shuffle ----------------
def predict_any(m, df: pd.DataFrame):
    if hasattr(m, "predict"):
        return m.predict(df)
    raise ValueError("Model non compatibile con predict")

def probe_values_for_feature(series: pd.Series, n_probes=5):
    if pd.api.types.is_numeric_dtype(series):
        return np.linspace(series.min(), series.max(), n_probes)
    else:
        return series.dropna().unique()

def model_sensitivity_given_Z(m, df: pd.DataFrame, Y: str, X: str, Z: List[str]) -> float:
    base = df.dropna(subset=[c for c in [X] + Z if c in df.columns]).copy()
    if base.empty:
        return np.nan
    probes = probe_values_for_feature(base[X])
    if len(probes) <= 1:
        return 0.0
    means = []
    for v in probes:
        temp = base.copy()
        temp.loc[:, X] = v
        preds = predict_any(m, temp)
        means.append(float(np.mean(preds)))
    return float(np.max(means) - np.min(means))

def shuffle_within_bins(df: pd.DataFrame, col: str, bin_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    if not bin_cols:
        out.loc[:, col] = _rng.permutation(out[col].values)
        return out
    for _, g in out.groupby(bin_cols, dropna=False):
        idx = g.index
        out.loc[idx, col] = _rng.permutation(out.loc[idx, col].values)
    return out

def shuffle_delta(m, df: pd.DataFrame, Y: str, X: str, parent_cols: List[str]) -> float:
    needed = [c for c in [X] + parent_cols if c in df.columns]
    base = df.dropna(subset=needed).copy()
    if base.empty:
        return np.nan
    p0 = predict_any(m, base)
    broken = shuffle_within_bins(base, X, parent_cols)
    p1 = predict_any(m, broken)
    return float(np.mean(np.abs(p0 - p1)))

# ---------------- Lettura dati ----------------
df = pd.read_csv("_Data/data_1.csv").dropna()

target_nodes = ["PDI", "SIZE"]

dag_edges = [('FRR', 'AQUEOUS'), ('AQUEOUS', 'PEG'), ('CHOL', 'ESM'), ('ESM', 'HSPC'), 
             ('AQUEOUS', 'TFR'), ('PEG', 'PDI'), ('PEG', 'CHOL'), ('ESM', 'SIZE'), 
             ('ESM', 'PDI'), ('TFR', 'PDI'), ('FRR', 'PDI'), ('TFR', 'SIZE'), ('FRR', 'SIZE')]
dag = nx.DiGraph(dag_edges)

model_path = "_Model/refined_model_size_pdi.pkl"
with open(model_path, "rb") as f:
    model_ = pickle.load(f)

modello_rf = model_["rf_model"]


# ---------------- Leggi file ACE ----------------
with open("6_VALIDAZIONE_INTERVENTISTA/_log/AverageCausalEffect_DAG2.log", "r") as f:
    lines = f.readlines()

pattern = r"ACE generale\((\w+) -> (\w+)\) = ([\d\.\-eE]+)"
records = []
for line in lines:
    match = re.search(pattern, line)
    if match:
        X, Y, value = match.groups()
        records.append({"X": X, "Y": Y, "ACE": float(value)})
df_ace = pd.DataFrame(records)

# ---------------- Leggi file SHAP ----------------
shap_log_file = "6_VALIDAZIONE_INTERVENTISTA/_log/shap_size_pdi.log"
shap_records = []
with open(shap_log_file, "r", encoding="utf-8") as f:
    lines = f.readlines()[4:]  # salta header
    for line in lines:
        if line.strip() == "":
            continue
        parts = line.split("|")
        if len(parts) == 3:
            target = parts[0].strip()
            feature = parts[1].strip()
            shap_value = float(parts[2].strip())
            shap_records.append({"Target": target, "Feature": feature, "MeanAbsSHAP": shap_value})
df_shap = pd.DataFrame(shap_records)

# ---------------- Sensitivity & Shuffle ----------------
results = []

for y in target_nodes:
    for x in df.columns:
        if x == y:
            continue
        parents = list(dag.predecessors(x))
        
        sens = model_sensitivity_given_Z(modello_rf, df, Y=y, X=x, Z=parents)
        shuf = shuffle_delta(modello_rf, df, Y=y, X=x, parent_cols=parents)

        # Trova ACE corrispondente
        ace_row = df_ace[(df_ace["X"]==x) & (df_ace["Y"]==y)]
        ace_value = ace_row["ACE"].values[0] if not ace_row.empty else 0.0

        # Trova SHAP corrispondente
        shap_row = df_shap[(df_shap["Target"]==y) & (df_shap["Feature"]==x)]
        shap_value = shap_row["MeanAbsSHAP"].values[0] if not shap_row.empty else 0.0

        results.append({
            "X": x,
            "Y": y,
            "Parents(Z)": ",".join(parents) if parents else "None",
            "Sensitivity": sens,
            "ShuffleDelta": shuf,
            "Sensitivity_scaled": sens,
            "ShuffleDelta_scaled": shuf,
            "ACE": ace_value,
            "MeanAbsSHAP": shap_value
        })

res = pd.DataFrame(results)

# ---------------- Percentili e spurious flag ----------------
res["ace_abs"] = res["ACE"].abs()
res["ace_pct"] = res["ace_abs"].rank(pct=True)
res["sens_pct"] = res["Sensitivity_scaled"].rank(pct=True)
res["shuffle_pct"] = res["ShuffleDelta_scaled"].rank(pct=True)
res["shap_pct"] = res["MeanAbsSHAP"].rank(pct=True)

min= 0.3
max=0.7
# Flag spurious: piccolo ACE ma grande Sensitivity/Shuffle o SHAP alto
res["spurious_flag_ace"] = (
    ((res["ace_pct"] <= min) & ((res["sens_pct"] >= max) | (res["shuffle_pct"] >= max))) 
)


res["spurious_flag_ace_shap"] = (
    (((res["ace_pct"] <= min) & (res["shap_pct"] >= max))) 
)

res["spurious_flag_shap"] = (
    (((res["shap_pct"] >= max) & (res["sens_pct"] <= min)) | ((res["shuffle_pct"] <= min))) 
)

# ---------------- Salva log globale ----------------
log_file_path = "6_VALIDAZIONE_INTERVENTISTA/_log/spurious_analysis.log"
with open(log_file_path, "w", encoding="utf-8") as log_f:
    
    # 1. ACE generali
    log_f.write("### ACE GENERALI ###\n")
    log_f.write(f"{'X':<12} | {'Y':<12} | {'ACE':>12}\n")
    log_f.write(f"{'-'*12}-|-{'-'*12}-|-{'-'*12}\n")
    for idx, row in df_ace.iterrows():
        log_f.write(f"{row['X']:<12} | {row['Y']:<12} | {row['ACE']:>12.6f}\n")
    
           
    # 4. Confronto ACE vs SHAP
    log_f.write("\n### CONFRONTO ACE vs SHAP ###\n")
    log_f.write(f"{'X':<12} | {'Y':<12} | {'ACE':>12} | {'MeanAbsSHAP':>12} | {'spurious_flag_shap':>8}\n")
    log_f.write(f"{'-'*12}-|-{'-'*12}-|-{'-'*12}-|-{'-'*12}-|-{'-'*16}-\n")
    for idx, row in res.iterrows():
        log_f.write(
            f"{row['X']:<12} | {row['Y']:<12} | {row['ACE']:>12.6f} | {row['MeanAbsSHAP']:>12.6f} | "
            f"{str(row['spurious_flag_shap']):>8}\n"
        )

    
    # 2. Euristica spurious flag (Sensitivity & Shuffle)
    log_f.write("\n### EURISTICA SENSITIVITY & SHUFFLE ###\n")
    log_f.write(f"{'X':<12} | {'Y':<12} | {'Parents(Z)':<20} | {'Sensitivity':>12} | {'ShuffleDelta':>12} | {'Spurious_ace':>8} | {'Spurious_shap':>8}\n")
    log_f.write(f"{'-'*12}-|-{'-'*12}-|-{'-'*20}-|-{'-'*12}-|-{'-'*12}-|-{'-'*16}|-{'-'*16}\n")
    for idx, row in res.iterrows():
        log_f.write(
            f"{row['X']:<12} | {row['Y']:<12} | {row['Parents(Z)']:<20} | "
            f"{row['Sensitivity']:>12.6f} | {row['ShuffleDelta']:>12.6f} | "
            f"{str(row['spurious_flag_ace']):>8} | "
            f"{str(row['spurious_flag_shap']):>8}\n"
        )


 
print(f"Log tabellare completo salvato in: {log_file_path}")
