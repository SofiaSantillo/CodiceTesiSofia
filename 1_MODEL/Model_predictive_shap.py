import os
import sys
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
from xgboost_gridsearch_pdi import functionalize_xgboost_gridsearch_pdi
from xgboost_gridsearch_size import functionalize_xgboost_gridsearch_size
from random_forest_size_pdi import random_forest_size_pdi
import seaborn as sns
import shap

sys.stdout = open("_Logs/analisi_shap.log", "w")


if __name__=="__main__":
    max_iter=10000
    threshold_size=0.85
    threshold_pdi=0.75

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from _File.config import data_path

    file_path = data_path + "/seed_non_ordinato.csv"
    DATA = pd.read_csv(file_path)
    DATA.dropna(inplace=True)

    r2_size_history = []
    r2_pdi_history = []
    
    rf_r2_pdi, rf_r2_size, data, rf_model= random_forest_size_pdi(DATA=DATA)
    best_rf_r2_size = rf_r2_size
    best_rf_r2_pdi = rf_r2_pdi

    r2_size_history.append(best_rf_r2_size)
    r2_pdi_history.append(best_rf_r2_pdi)
      

    for i in range(max_iter):
        if best_rf_r2_size >= threshold_size and best_rf_r2_pdi >= threshold_pdi:
            print(f"Early stopping at iteration {i+1} due to insufficient improvement.")
            break
        if(best_rf_r2_pdi > best_rf_r2_size): 
            data = functionalize_xgboost_gridsearch_pdi(data=data)
            data = functionalize_xgboost_gridsearch_size(data=data)
        else:   
            data = functionalize_xgboost_gridsearch_size(data=data)
            data = functionalize_xgboost_gridsearch_pdi(data=data)

        rf_r2_pdi, rf_r2_size, data, model_rf= random_forest_size_pdi(data)
            
        best_rf_r2_pdi = rf_r2_pdi
        best_rf_r2_size = rf_r2_size

        r2_size_history.append(best_rf_r2_size)
        r2_pdi_history.append(best_rf_r2_pdi)
 


    plt.figure(figsize=(10, 5))
    plt.plot(r2_size_history, label='R² SIZE', marker='o')
    plt.plot(r2_pdi_history, label='R² PDI', marker='x')
    plt.xlabel('Iterazione')
    plt.ylabel('R²')
    plt.title('Evoluzione di R² per SIZE e PDI')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('_Plot/r2_evolution.png') 
    plt.show()

    X = data.drop(columns=["SIZE", "PDI"])   


    # Predizioni dal modello finale
    y_pred = model_rf.predict(X)

    # Creazione di un DataFrame con le predizioni e i nomi delle colonne
    predictions_df = pd.DataFrame(y_pred, columns=["SIZE_pred", "PDI_pred"])


    predictions_df.to_csv("6_ANALISI_INTERVENTISTA/_csv/Predizione_ML_originaria.csv", index=False)

####### ANALISI SHAP ######### 
X = data.drop(columns=['SIZE', 'PDI'])

if 'preprocessor' in rf_model.named_steps:
    preprocessor = rf_model.named_steps['preprocessor']
    X_transformed = preprocessor.transform(X)
    feature_names_transformed = preprocessor.get_feature_names_out()
else:
    X_transformed = X.values
    feature_names_transformed = X.columns

rf_model_size = rf_model.named_steps['regressor'].estimators_[0]
rf_model_pdi  = rf_model.named_steps['regressor'].estimators_[1]

feature_map = {
    "cat__ML_ESM": "ML",
    "cat__ML_HSPC": "ML",
    "cat__CHIP_Droplet": "CHIP",
    "cat__CHIP_Micromixer": "CHIP",
    "cat__OUTPUT_YES": "OUTPUT",
    "cat__OUTPUT_NO": "OUTPUT",
    "cat__AQUEOUS_MQ": "AQUEOUS",
    "cat__AQUEOUS_PBS": "AQUEOUS",
    "remainder__ESM": "ESM",
    "remainder__CHOL": "CHOL",
    "remainder__FRR": "FRR",
    "remainder__TFR": "TFR",
    "remainder__HSPC": "HSPC",
    "remainder__PEG": "PEG",
    # aggiungi altre feature categoriali se presenti
}

unique_features = list(set(feature_map.values()))

# SHAP per SIZE
explainer_size = shap.TreeExplainer(rf_model_size)
shap_values_size = explainer_size.shap_values(X_transformed)

# SHAP per PDI
explainer_pdi = shap.TreeExplainer(rf_model_pdi)
shap_values_pdi = explainer_pdi.shap_values(X_transformed)

def aggregate_shap(shap_vals, feature_names_transformed, feature_map):
    unique_features = list(set(feature_map.values()))
    shap_agg = np.zeros((shap_vals.shape[0], len(unique_features)))
    
    for i, col in enumerate(feature_names_transformed):
        orig_feat = feature_map.get(col, col)
        idx = unique_features.index(orig_feat)
        shap_agg[:, idx] += shap_vals[:, i]
    return shap_agg, unique_features

print(feature_names_transformed)
print(feature_map)
shap_values_size_agg, agg_features = aggregate_shap(shap_values_size, feature_names_transformed, feature_map)
shap_values_pdi_agg, _ = aggregate_shap(shap_values_pdi, feature_names_transformed, feature_map)

X_agg_df = pd.DataFrame(columns=agg_features, index=range(X_transformed.shape[0]))
for feat in agg_features:
    cols_to_sum = [c for c, f in feature_map.items() if f == feat]
    if cols_to_sum:
        idx_cols = [np.where(feature_names_transformed == c)[0][0] for c in cols_to_sum]
        X_agg_df[feat] = X_transformed[:, idx_cols].sum(axis=1)
    else:
        idx_col = np.where(feature_names_transformed == feat)[0][0]
        X_agg_df[feat] = X_transformed[:, idx_col]


# Summary plot SIZE
shap.summary_plot(shap_values_size_agg, X_agg_df, plot_type="bar", show=False)
plt.title("Feature importance per SIZE")
plt.tight_layout()
plt.savefig('_Plot/shap_summary_size.png')
plt.show()

# Summary plot PDI
shap.summary_plot(shap_values_pdi_agg, X_agg_df, plot_type="bar", show=False)
plt.title("Feature importance per PDI")
plt.tight_layout()
plt.savefig('_Plot/shap_summary_pdi.png')
plt.show()


# Importanza media delle feature aggregate per SIZE
shap_importance_size = np.abs(shap_values_size_agg).mean(axis=0)
shap_importance_size_df = pd.DataFrame({
    'Feature': agg_features,
    'Importance': shap_importance_size
}).sort_values(by='Importance', ascending=False)

print("Importanza media per SIZE:")
print(shap_importance_size_df)

# Importanza media delle feature aggregate per PDI
shap_importance_pdi = np.abs(shap_values_pdi_agg).mean(axis=0)
shap_importance_pdi_df = pd.DataFrame({
    'Feature': agg_features,
    'Importance': shap_importance_pdi
}).sort_values(by='Importance', ascending=False)

print("Importanza media per PDI:")
print(shap_importance_pdi_df)

