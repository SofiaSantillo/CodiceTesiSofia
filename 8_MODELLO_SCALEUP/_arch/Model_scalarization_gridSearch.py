import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from itertools import product

model_file = "_Model/refined_model_size_pdi.pkl"
with open(model_file, "rb") as f:
    models = pickle.load(f)

rf_model = models["rf_model"]
xgb_size = models["xgb_size"]
xgb_pdi = models["xgb_pdi"]

val_df = pd.read_csv("_Data/dataset_ScaleUp.csv").dropna()
X_val = val_df.drop(columns=["SIZE", "PDI"])
y_val_size = val_df["SIZE"]
y_val_pdi = val_df["PDI"]

numeric_cols = X_val.select_dtypes(include=np.number).columns.tolist()
exclude_cols = ["AQUEOUS", "CHOL", "HSPC", "FRR"]
cols_to_scale = [col for col in numeric_cols if col not in exclude_cols]

def iterative_refinement_predict(X_raw, rf_model, xgb_size, xgb_pdi, num_epochs=5):
    X = X_raw.copy()
    preds = rf_model.predict(X)
    size_pred = preds[:,0].ravel()
    pdi_pred = preds[:,1].ravel()
    for _ in range(num_epochs):
        X_with_pdi = X.copy()
        X_with_pdi["PDI"] = pdi_pred
        size_pred = xgb_size.predict(X_with_pdi).ravel()
        X_with_size = X.copy()
        X_with_size["SIZE"] = size_pred
        pdi_pred = xgb_pdi.predict(X_with_size).ravel()
    return pd.DataFrame({"Refined_SIZE": size_pred, "Refined_PDI": pdi_pred})

def compute_metrics(y_true_size, y_true_pdi, y_pred_df):
    return {
        "R2_SIZE": r2_score(y_true_size, y_pred_df["Refined_SIZE"]),
        "R2_PDI": r2_score(y_true_pdi, y_pred_df["Refined_PDI"]),
        "R2_mean": (r2_score(y_true_size, y_pred_df["Refined_SIZE"]) +
                    r2_score(y_true_pdi, y_pred_df["Refined_PDI"])) / 2
    }

scale_ranges = {
    "TFR": {"0-3": np.round(np.arange(0.101, 1.2, 0.1), 2), "3-25": np.round(np.arange(0.01, 0.97, 0.1), 2), "25-55": np.round(np.arange(0.001, 0.1, 0.1), 3)},
    "PEG": {"0-3": np.round(np.arange(0.01, 10.2, 1), 2), "3-6": np.round(np.arange(0.01, 0.97, 0.1), 2)},
    "ESM": {"0-3": np.round(np.arange(0.01, 10.1, 1), 2), "15-25": np.round(np.arange(0.60, 1.1, 0.1), 2)},
    "FRR": {"0-5": np.round(np.arange(0.1, 3.1, 0.1), 2), "5-10": np.round(np.arange(0.80, 1.8, 0.1), 2)}
}

def generate_cluster_combinations(scale_ranges, cols_to_scale):
    col_combinations = []
    col_names = []
    for col in cols_to_scale:
        if col in scale_ranges:
            factors_list = [f for cluster in scale_ranges[col].values() for f in cluster]
            col_combinations.append(factors_list)
            col_names.append(col)
    return col_names, list(product(*col_combinations))

def apply_combination(X, col_names, factor_comb):
    X_scaled = X.copy()
    for col, f in zip(col_names, factor_comb):
        X_scaled[col] = X[col] * f
    return X_scaled

col_names, all_combinations = generate_cluster_combinations(scale_ranges, cols_to_scale)
print(len(all_combinations))
best_r2 = -np.inf
best_combination = None
i=0
for comb in all_combinations:
    print(i)
    i=i+1
    X_scaled = apply_combination(X_val, col_names, comb)
    y_pred = iterative_refinement_predict(X_scaled, rf_model, xgb_size, xgb_pdi, num_epochs=5)
    r2_mean = compute_metrics(y_val_size, y_val_pdi, y_pred)["R2_mean"]
    if r2_mean > best_r2:
        best_r2 = r2_mean
        best_combination = comb

X_scaled_final = apply_combination(X_val, col_names, best_combination)
y_pred_final = iterative_refinement_predict(X_scaled_final, rf_model, xgb_size, xgb_pdi, num_epochs=5)
final_metrics = compute_metrics(y_val_size, y_val_pdi, y_pred_final)

print("Combinazione globale ottimale dei fattori di scala:")
for col, f in zip(col_names, best_combination):
    print(f"{col}: {f}")
print("\nMetriche finali:")
for k, v in final_metrics.items():
    print(f"{k}: {v:.4f}")
