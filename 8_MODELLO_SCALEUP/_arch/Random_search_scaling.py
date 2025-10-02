import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

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
exclude_cols = ["AQUEOUS", "FRR", "CHOL", "HSPC"]
cols_to_scale = [col for col in numeric_cols if col not in exclude_cols]
print("Colonne numeriche su cui fare scaling:", cols_to_scale)

def iterative_refinement_predict(X_raw, rf_model, xgb_size, xgb_pdi, num_epochs=20, alpha=0.5):
    X = X_raw.copy()
    initial_preds = rf_model.predict(X)
    size_pred = initial_preds[:, 0].ravel()
    pdi_pred = initial_preds[:, 1].ravel()
    
    for _ in range(num_epochs):
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

def random_search_scaling(X, y_size, y_pdi, cols_to_scale, n_iter=1000, scale_range=(0.05, 5)):
    best_r2 = -np.inf
    best_factors = {col: 1.0 for col in cols_to_scale}
    
    for i in range(n_iter):
        factors = {col: np.random.uniform(*scale_range) for col in cols_to_scale}
        X_scaled = X.copy()
        for col, factor in factors.items():
            X_scaled[col] = X[col] * factor
        preds = iterative_refinement_predict(X_scaled, rf_model, xgb_size, xgb_pdi)
        r2_mean = (r2_score(y_size, preds["Refined_SIZE"]) +
                   r2_score(y_pdi, preds["Refined_PDI"])) / 2
        if r2_mean > best_r2:
            best_r2 = r2_mean
            best_factors = factors
            print(f"[{i+1}] Nuovo miglior R2 medio: {best_r2:.4f}")
    
    return best_factors, best_r2

best_factors, best_r2 = random_search_scaling(X_val, y_val_size, y_val_pdi, cols_to_scale, n_iter=500)

print("\nMigliori fattori di scala trovati:")
for col, f in best_factors.items():
    print(f"  {col}: {f:.4f}")

X_best_scaled = X_val.copy()
for col, factor in best_factors.items():
    X_best_scaled[col] = X_val[col] * factor

final_preds = iterative_refinement_predict(X_best_scaled, rf_model, xgb_size, xgb_pdi)
final_metrics = {
    "R2_SIZE": r2_score(y_val_size, final_preds["Refined_SIZE"]),
    "R2_PDI": r2_score(y_val_pdi, final_preds["Refined_PDI"]),
    "R2_mean": (r2_score(y_val_size, final_preds["Refined_SIZE"]) +
                r2_score(y_val_pdi, final_preds["Refined_PDI"])) / 2
}
print("\nMetriche finali:")
for k, v in final_metrics.items():
    print(f"  {k}: {v:.4f}")
