import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error



class RefinedModel:
    def __init__(self, base_estimator, regressor_size, regressor_pdi, num_epochs=5):
        self.base_estimator = base_estimator
        self.regressor_size = regressor_size
        self.regressor_pdi = regressor_pdi
        self.num_epochs = num_epochs

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        X = X.copy()
        initial_preds = self.base_estimator.predict(X)
        size_pred = initial_preds[:, 0].ravel()
        pdi_pred = initial_preds[:, 1].ravel()

        for _ in range(self.num_epochs):
            X_with_pdi = X.copy()
            X_with_pdi["PDI"] = pdi_pred
            size_pred = self.regressor_size.predict(X_with_pdi).ravel()

            X_with_size = X.copy()
            X_with_size["SIZE"] = size_pred
            pdi_pred = self.regressor_pdi.predict(X_with_size).ravel()

        return pd.DataFrame({
            "Refined_SIZE": size_pred,
            "Refined_PDI": pdi_pred
        })

model_file = "_Model/refined_model_size_pdi.pkl"
with open(model_file, "rb") as f:
    models = pickle.load(f)

rf_model = models["rf_model"]
xgb_size = models["xgb_size"]
xgb_pdi = models["xgb_pdi"]

refined_model = RefinedModel(
    base_estimator=rf_model,
    regressor_size=xgb_size,
    regressor_pdi=xgb_pdi,
    num_epochs=5
)

val_df = pd.read_csv("_Data/dataset_ScaleUp.csv").dropna()
X_val = val_df.drop(columns=["SIZE", "PDI"])
y_val_size = val_df["SIZE"]
y_val_pdi = val_df["PDI"]

y_pred_df = refined_model.predict(X_val)

for col, y_true in zip(["SIZE", "PDI"], [y_val_size, y_val_pdi]):
    y_pred = y_pred_df[f"Refined_{col}"]
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{col} → MSE: {mse:.4f}, R2: {r2:.4f}")