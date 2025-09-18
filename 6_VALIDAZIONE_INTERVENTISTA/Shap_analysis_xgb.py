import os
import pickle
import sys
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------------
# 1. Caricamento modelli e dati
# -----------------------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from _File.config import models_path, data_path

model_file = os.path.join(models_path, "refined_model_size_pdi.pkl")
dataset_file = os.path.join(data_path, "data_1.csv")

with open(model_file, "rb") as f:
    models_dict = pickle.load(f)

# Modello che predice solo PDI
model = models_dict['xgb_pdi']

data = pd.read_csv(dataset_file).dropna()
X_raw = data.copy()

log_path = "6_VALIDAZIONE_INTERVENTISTA/_log/"
plot_path = "6_VALIDAZIONE_INTERVENTISTA/_plot/"

# -----------------------------------------------------------------------------------
# 4. Analisi SHAP
# -----------------------------------------------------------------------------------
X_preprocessed = model.named_steps['preprocessor'].transform(X_raw)

def rf_predict_wrapper(X):
    return model.named_steps['regressor'].predict(X)

explainer = shap.Explainer(rf_predict_wrapper, X_preprocessed)
shap_values = explainer(X_preprocessed)

# -----------------------------------------------------------------------------------
# 5. Creazione log SHAP - versione nuova
# -----------------------------------------------------------------------------------

out_folder = "6_VALIDAZIONE_INTERVENTISTA/_log"
log_ref_path = f"{out_folder}/sensitivity_shuffle.log"
df_log = pd.read_csv(log_ref_path, sep="|")
df_log.columns = [c.strip() for c in df_log.columns]  # pulizia spazi
df_log['SHAP'] = np.nan 

# Calcolo mean absolute SHAP solo per SIZE
mean_abs_shap_pdi = np.abs(shap_values.values).mean(axis=0)

# Nome delle feature preprocessate
if hasattr(model.named_steps['preprocessor'], 'get_feature_names_out'):
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
else:
    feature_names = X_raw.columns.tolist()

# Creazione dataframe SHAP (solo SIZE)
df_shap = pd.DataFrame({
    'Target': ['PDI']*len(feature_names),
    'Feature': list(feature_names),
    'SHAP_Value': mean_abs_shap_pdi
})

df_shap['Feature'] = df_shap['Feature'] \
    .str.replace('cat__', '', regex=False) \
    .str.replace('remainder__', '', regex=False) \
    .str.replace('_PBS', '', regex=False)

# ---------------- Aggiornamento df_log con SHAP ----------------
df_log['SHAP'] = np.nan

for idx, row in df_log.iterrows():
    target = row['Target'].strip()
    feature = row['Feature'].strip()
    
    # Match tra target e feature
    match = df_shap[(df_shap['Target'] == target) & (df_shap['Feature'].str.strip() == feature)]
    if not match.empty:
        df_log.at[idx, 'SHAP'] = match['SHAP_Value'].values[0]

# ---------------- Scrittura su log in forma tabellare ----------------
log_file_shap = f"{out_folder}/Complete_Analysis.log"

with open(log_file_shap, "w") as f:
    f.write(f"{'Target':<13}| {'Feature':<19}| {'DAG_Category':<15}| "
            f"{'TCE_Value':<14}| {'Percentuale':<14}| {'Sensitivity':<13}| "
            f"{'ShuffleDelta':<13}| {'SHAP':<12}\n")
    
    for idx, row in df_log.iterrows():
        target = row['Target']
        feature = row['Feature']
        category = row['DAG_Category']
        tce = row['TCE_Value']
        pct = row['Percentuale']
        sens = row['Sensitivity']
        shuf = row['Shuffle']
        
        shap_val = "" if pd.isna(row['SHAP']) else f"{row['SHAP']:.6f}"
        
        f.write(f"{row['Target']:<12}| {row['Feature']:<17}| {row['DAG_Category']:<14}| "
                f"{tce:<13}| {pct:<13}| {sens:<12}| {shuf:<12}| {shap_val:<12}\n")

print(f"Log scritto in {log_file_shap}")
