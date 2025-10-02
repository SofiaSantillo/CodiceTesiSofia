import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

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

def iterative_refinement_predict(X_raw, rf_model, xgb_size, xgb_pdi, num_epochs=5):
    X = X_raw.copy()
    initial_preds = rf_model.predict(X)
    size_pred = initial_preds[:, 0].ravel()
    pdi_pred = initial_preds[:, 1].ravel()
    
    for epoch in range(num_epochs):
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

y_pred_df = iterative_refinement_predict(X_val, rf_model, xgb_size, xgb_pdi, num_epochs=5)

for col, y_true in zip(["SIZE", "PDI"], [y_val_size, y_val_pdi]):
    y_pred = y_pred_df[f"Refined_{col}"]
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{col} → MSE: {mse:.4f}, R2: {r2:.4f}")