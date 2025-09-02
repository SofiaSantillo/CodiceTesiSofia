from matplotlib import pyplot as plt
import pandas as pd
import shap
from random_forest_size_pdi import random_forest_size_pdi
import os
import numpy as np

# --- Carica dati ---
file_path = "Data_Droplet/seed_non_ordinato.csv"
DATA = pd.read_csv(file_path)
DATA.dropna(inplace=True)

# --- Esegui Random Forest ---
val1, val2, data, model_pipeline = random_forest_size_pdi(DATA=DATA)

# --- Estrai il MultiOutputRegressor dalla pipeline ---
multi_rf_model = model_pipeline.named_steps['regressor']
model_size = multi_rf_model.estimators_[0]
model_pdi  = multi_rf_model.estimators_[1]

# --- Mantieni tutte le colonne per la trasformazione ---
X_full = data.drop(columns=['SIZE', 'PDI'], errors='ignore')

# --- Trasforma le feature tramite il preprocessor della pipeline ---
preprocessor = model_pipeline.named_steps['preprocessor']
X_transformed = preprocessor.transform(X_full)

# --- Nomi feature trasformate ---
cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out()
num_features = [col for col in X_full.columns if col not in preprocessor.transformers_[0][2]]
feature_names = list(cat_features) + num_features

# --- Directory per i file PNG ---
os.makedirs('_Plot', exist_ok=True)

# --- Funzione per SHAP plots ---
def shap_all_plots(model, X_transformed, feature_names, target_name):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)

    # --- Summary plot ---
    shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False)
    plt.title(f"Summary plot per {target_name}")
    plt.tight_layout()
    plt.savefig(f'_Plot/shap_summary_{target_name}.png')
    plt.close()

    # --- Bar plot ---
    shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, plot_type="bar", show=False)
    plt.title(f"Bar plot per {target_name}")
    plt.tight_layout()
    plt.savefig(f'_Plot/shap_bar_{target_name}.png')
    plt.close()

    # --- Dependence plot per feature più importante ---
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = mean_abs_shap.argmax()
    top_feature = feature_names[top_idx]

    shap.dependence_plot(top_feature, shap_values, X_transformed, feature_names=feature_names, show=False)
    plt.title(f"Dependence plot {top_feature} per {target_name}")
    plt.tight_layout()
    plt.savefig(f'_Plot/shap_dependence_{target_name}_{top_feature}.png')
    plt.close()

# --- SHAP plots per SIZE ---
shap_all_plots(model_size, X_transformed, feature_names, "SIZE")

# --- SHAP plots per PDI ---
shap_all_plots(model_pdi, X_transformed, feature_names, "PDI")
