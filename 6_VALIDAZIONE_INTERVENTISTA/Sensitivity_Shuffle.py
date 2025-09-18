import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# ---------------- RNG per shuffle ----------------
_rng = np.random.default_rng(42)
out_folder = "6_VALIDAZIONE_INTERVENTISTA/_log"
log_ref_path = "6_VALIDAZIONE_INTERVENTISTA/_log/TCE.log"
df_ref = pd.read_csv(log_ref_path, sep="|")
df_ref.columns = [c.strip() for c in df_ref.columns]

# ---------------- Funzioni Sensitivity / Shuffle ----------------
def predict_any(model, df: pd.DataFrame):
    """Predizione generica compatibile con sklearn-like models."""
    if hasattr(model, "predict"):
        return model.predict(df)
    raise ValueError("Model non compatibile con predict")

def probe_values_for_feature(series: pd.Series, n_probes=5):
    """Valori da testare per sensitivity (numeric o categorical)."""
    if pd.api.types.is_numeric_dtype(series):
        return np.linspace(series.min(), series.max(), n_probes)
    else:
        return series.dropna().unique()

def model_sensitivity(model, df: pd.DataFrame, X: str) -> float:
    """Sensibilità globale della feature X senza considerare DAG."""
    base = df.dropna(subset=[X]).copy()
    if base.empty:
        return np.nan
    probes = probe_values_for_feature(base[X])
    if len(probes) <= 1:
        return 0.0
    means = []
    for v in probes:
        temp = base.copy()
        temp[X] = v
        preds = predict_any(model, temp)
        means.append(np.mean(preds))
    return float(np.max(means) - np.min(means))

def shuffle_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Shuffle globale della feature col."""
    out = df.copy()
    out[col] = _rng.permutation(out[col].values)
    return out

def shuffle_delta(model, df: pd.DataFrame, X: str) -> float:
    """Importanza della feature X basata sul peggioramento delle predizioni dopo shuffle."""
    base = df.dropna(subset=[X]).copy()
    if base.empty:
        return np.nan
    p0 = predict_any(model, base)
    broken = shuffle_column(base, X)
    p1 = predict_any(model, broken)
    return float(np.mean(np.abs(p0 - p1)))

# ---------------- Lettura dati ----------------
df = pd.read_csv("_Data/data_1.csv").dropna()
target_nodes = ["PDI", "SIZE"]

model_path = "_Model/refined_model_size_pdi.pkl"
with open(model_path, "rb") as f:
    model_ = pickle.load(f)

modello_rf = model_["xgb_pdi"]

# ---------------- Sensitivity & Shuffle senza DAG ----------------
results = []
df_log = df_ref.copy()
df_log['Sensitivity'] = np.nan
df_log['ShuffleDelta'] = np.nan

for y in target_nodes:
    for x in df.columns:
        if x == y:
            continue  # non usare il target come feature

        sens = model_sensitivity(modello_rf, df, X=x)
        shuf = shuffle_delta(modello_rf, df, X=x)

        results.append({
            "Feature": x,
            "Target": y,
            "Sensitivity": sens,
            "ShuffleDelta": shuf
        })

res = pd.DataFrame(results)

# ---------------- Calcolo Sensitivity e Shuffle e aggiornamento df_log ----------------
for idx, row in df_log.iterrows():
    feature = row['Feature'].strip()
    target = row['Target'].strip()
    
    if feature not in df.columns or feature == target:
        continue
    
    sens = model_sensitivity(modello_rf, df, X=feature)
    shuf = shuffle_delta(modello_rf, df, X=feature)
    
    df_log.at[idx, 'Sensitivity'] = sens
    df_log.at[idx, 'ShuffleDelta'] = shuf

# ---------------- Scrittura su log in forma tabellare ----------------
log_file = f"{out_folder}/sensitivity_shuffle.log"
with open(log_file, "w") as f:
    f.write(f"{'Target':<13}| {'Feature':<18}| {'DAG_Category':<14}| {'TCE_Value':<13}| {'Percentuale':<13}| {'Sensitivity':<12}| {'Shuffle':<12}\n")
    for idx, row in df_log.iterrows():
        target = row['Target']
        feature = row['Feature']
        category = row['DAG_Category']
        tce = row['TCE_Value']
        pct = row['Percentuale']
        sens = "" if pd.isna(row['Sensitivity']) else f"{row['Sensitivity']:.6f}"
        shuf = "" if pd.isna(row['ShuffleDelta']) else f"{row['ShuffleDelta']:.6f}"
        f.write(f"{target:<12}| {feature:<12}| {category:<12}| {tce:<12}| {pct:<12}| {sens:<12}| {shuf:<12}\n")

print(f"Log scritto in {log_file}")
