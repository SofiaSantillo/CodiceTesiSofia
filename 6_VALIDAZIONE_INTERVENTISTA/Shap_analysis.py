import os
import sys
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------------
# 1. Configurazioni e percorsi
# -----------------------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from _File.config import models_path, data_path

model_file = os.path.join(models_path, "refined_model_size_pdi.pkl")
dataset_file = os.path.join(data_path, "data_1.csv")

with open(model_file, "rb") as f:
    models_dict = pickle.load(f)

rf_model = models_dict['rf_model']
xgb_size = models_dict['xgb_size']
xgb_pdi = models_dict['xgb_pdi']

data = pd.read_csv(dataset_file).dropna()
X_raw = data.copy()

log_path = "6_VALIDAZIONE_INTERVENTISTA/_log/"
plot_path = "6_VALIDAZIONE_INTERVENTISTA/_plot/"

# -----------------------------------------------------------------------------------
# 2. Preprocessing compatibile con SHAP
# -----------------------------------------------------------------------------------
def preprocess_pipeline_for_shap(pipeline, X_raw):
    preprocessor = pipeline.named_steps['preprocessor']
    X_transformed = preprocessor.transform(X_raw)

    feature_names = []
    for name, trans, cols in preprocessor.transformers_:
        if trans == 'passthrough':
            if isinstance(cols, slice):
                feature_names += list(X_raw.columns[cols])
            elif all(isinstance(c, int) for c in cols):
                feature_names += list(X_raw.columns[cols])
            else:
                feature_names += list(cols)
        else:
            if hasattr(trans, 'get_feature_names_out'):
                feature_names += list(trans.get_feature_names_out())

    return pd.DataFrame(X_transformed, columns=feature_names)

X_size_df = preprocess_pipeline_for_shap(xgb_size, X_raw)
X_pdi_df = preprocess_pipeline_for_shap(xgb_pdi, X_raw)

# -----------------------------------------------------------------------------------
# 3. Analisi SHAP e log unico
# -----------------------------------------------------------------------------------
def shap_analysis_combined(models, X_dfs, targets, log_file_path, plot_path):
    all_results = []

    for model_pipeline, X_df, target in zip(models, X_dfs, targets):
        explainer = shap.Explainer(model_pipeline.named_steps['regressor'].estimators_[0], X_df)
        shap_values = explainer(X_df)
        
        mean_abs_shap = pd.DataFrame({
            'Target': target,
            'Feature': X_df.columns,
            'MeanAbsSHAP': np.abs(shap_values.values).mean(axis=0)
        })
        all_results.append(mean_abs_shap)

        # Salva plot separato
        plt.figure(figsize=(10,6))
        shap.summary_plot(shap_values, X_df, plot_type="bar", show=False, max_display=20)
        plot_file = os.path.join(plot_path, f"shap_summary_{target}.png")
        plt.savefig(plot_file, bbox_inches='tight')
        plt.close()
        print(f"[INFO] SHAP plot for {target} saved at {plot_file}")

    # Combina tutti i risultati in un unico DataFrame
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Scrivi log
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write("### SHAP Summary Combined - SIZE & PDI ###\n")
        f.write(f"{'Target':<6} | {'Feature':<30} | {'MeanAbsSHAP':>12}\n")
        f.write(f"{'-'*6}-|-{'-'*30}-|-{'-'*12}\n")
        for _, row in combined_df.iterrows():
            f.write(f"{row['Target']:<6} | {row['Feature']:<30} | {row['MeanAbsSHAP']:>12.6f}\n")

    print(f"[INFO] Combined SHAP log saved at {log_file_path}")

# Esegui funzione combinata
shap_analysis_combined(
    models=[xgb_size, xgb_pdi],
    X_dfs=[X_size_df, X_pdi_df],
    targets=["SIZE", "PDI"],
    log_file_path=os.path.join(log_path, "shap_size_pdi.log"),
    plot_path=plot_path
)
