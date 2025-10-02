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

def iterative_rf_xgb_with_separate_scale(X, rf_model, xgb_size, xgb_pdi,
                                         esm_factor, peg_factor, num_epochs=10):
    X_iter = X.copy()
    
    initial_preds = rf_model.predict(X_iter)
    size_pred = initial_preds[:, 0].ravel()
    pdi_pred = initial_preds[:, 1].ravel()
    
    for _ in range(num_epochs):
        X_size = X_iter.copy()
        X_size["ESM"] = X_iter["ESM"] * esm_factor
        X_size["PEG"] = X_iter["PEG"] 
        X_size["PDI"] = pdi_pred
        size_pred = xgb_size.predict(X_size).ravel()
        

        X_pdi = X_iter.copy()
        X_pdi["ESM"] = X_iter["ESM"] 
        X_pdi["PEG"] = X_iter["PEG"] * peg_factor
        X_pdi["SIZE"] = size_pred
        pdi_pred = xgb_pdi.predict(X_pdi).ravel()
    
    return size_pred, pdi_pred

scale_factors = np.round(np.arange(0.1, 10.01, 0.5), 2)
best_r2_mean = -np.inf
best_factor_ESM = 1.0
best_factor_PEG = 1.0
best_factor_TFR = 1.0

for esm_factor in scale_factors:
    for peg_factor in scale_factors:
        size_pred, pdi_pred = iterative_rf_xgb_with_separate_scale(
            X_val, rf_model, xgb_size, xgb_pdi, esm_factor, peg_factor, num_epochs=10
        )
        r2_size = r2_score(y_val_size, size_pred)
        r2_pdi = r2_score(y_val_pdi, pdi_pred)
        r2_mean = (r2_size + r2_pdi) / 2
            
        if r2_mean > best_r2_mean:
            best_r2_mean = r2_mean
            best_factor_ESM = esm_factor
            best_factor_PEG = peg_factor

print(f"\nMigliori fattori trovati: ESM -> {best_factor_ESM}, PEG -> {best_factor_PEG}, (R2 medio = {best_r2_mean:.4f})")

size_pred_final, pdi_pred_final = iterative_rf_xgb_with_separate_scale(
    X_val, rf_model, xgb_size, xgb_pdi, best_factor_ESM, best_factor_PEG, num_epochs=10
)

final_metrics = {
    "R2_SIZE": r2_score(y_val_size, size_pred_final),
    "R2_PDI": r2_score(y_val_pdi, pdi_pred_final),
    "R2_mean": (r2_score(y_val_size, size_pred_final) + r2_score(y_val_pdi, pdi_pred_final)) / 2,
    "MSE_SIZE": ((y_val_size - size_pred_final)**2).mean(),
    "MSE_PDI": ((y_val_pdi - pdi_pred_final)**2).mean(),
    "MAE_SIZE": (abs(y_val_size - size_pred_final)).mean(),
    "MAE_PDI": (abs(y_val_pdi - pdi_pred_final)).mean()
}

print("\n--- Metriche finali ---")
for k, v in final_metrics.items():
    print(f"{k}: {v:.4f}")
