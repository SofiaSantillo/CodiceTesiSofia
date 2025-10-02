import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score

# === Caricamento modelli ===
model_file = "_Model/refined_model_size_pdi.pkl"
with open(model_file, "rb") as f:
    models = pickle.load(f)

rf_model = models["rf_model"]
xgb_size = models["xgb_size"]
xgb_pdi = models["xgb_pdi"]

# === Caricamento dataset ===
val_df = pd.read_csv("_Data/dataset_ScaleUp.csv").dropna()
#val_df2 = pd.read_csv("8.1_MODELLO_SUP/dataset_sampling_DAG2.csv").dropna()
#val_df = pd.concat([val_df1, val_df2], ignore_index=True)

X_val = val_df.drop(columns=["SIZE", "PDI"])
y_val_size = val_df["SIZE"].values
y_val_pdi = val_df["PDI"].values

# === Funzione di predizione iterativa ===
def iterative_refinement_predict(X_raw, rf_model, xgb_size, xgb_pdi, num_epochs=10):
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

# === Predizioni base ===
refined_preds = iterative_refinement_predict(X_val, rf_model, xgb_size, xgb_pdi)
raw_size_pred = refined_preds["Refined_SIZE"].values
raw_pdi_pred = refined_preds["Refined_PDI"].values

print("R² SIZE (raw):", r2_score(y_val_size, raw_size_pred))
print("R² PDI (raw):", r2_score(y_val_pdi, raw_pdi_pred))

# === 1. Calibrazione Polynomial Regression ===
poly = PolynomialFeatures(degree=5, include_bias=False)
X_poly = poly.fit_transform(raw_size_pred.reshape(-1, 1))
poly_reg = LinearRegression().fit(X_poly, y_val_size)
poly_size_pred = poly_reg.predict(X_poly)

print("Polynomial Calibrated R² SIZE:", r2_score(y_val_size, poly_size_pred))

# === 2. Quantile Mapping (Isotonic Regression) ===
iso_reg = IsotonicRegression(out_of_bounds="clip")
iso_reg.fit(raw_size_pred, y_val_size)
iso_size_pred = iso_reg.predict(raw_size_pred)

print("Quantile Calibrated R² SIZE:", r2_score(y_val_size, iso_size_pred))
