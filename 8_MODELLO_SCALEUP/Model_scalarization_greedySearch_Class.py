import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import sys

class Logger(object):
    def __init__(self, filename="output.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)  
        self.log.write(message)       

    def flush(self):  
        pass

sys.stdout = Logger("8_MODELLO_SCALEUP/_log/Model_scalarization_greedSearch_Class.log")

# =============================
# 1. Classe RefinedModel
# =============================

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


# =============================
# 2. Caricamento modelli bundle
# =============================

print( "\n-------------  START MODEL SCALARIZATION GREED SEARCH (Class RefinedModel - no range) ------------------")
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
    num_epochs=10
)


# =============================
# 3. Dataset
# =============================
val_df = pd.read_csv("_Data/dataset_ScaleUp.csv").dropna()
X_val = val_df.drop(columns=["SIZE", "PDI"])
y_val_size = val_df["SIZE"]
y_val_pdi = val_df["PDI"]

numeric_cols = X_val.select_dtypes(include=np.number).columns.tolist()
exclude_cols = ["AQUEOUS", "FRR", "CHOL", "HSPC"]
cols_to_scale = [col for col in numeric_cols if col not in exclude_cols]
print("Colonne numeriche su cui fare scaling:", cols_to_scale)


# =============================
# 4. Metriche
# =============================
def compute_metrics(y_true_size, y_true_pdi, y_pred_df):
    return {
        "R2_SIZE": r2_score(y_true_size, y_pred_df["Refined_SIZE"]),
        "R2_PDI": r2_score(y_true_pdi, y_pred_df["Refined_PDI"]),
        "R2_mean": (r2_score(y_true_size, y_pred_df["Refined_SIZE"]) +
                    r2_score(y_true_pdi, y_pred_df["Refined_PDI"])) / 2,
        "MSE_SIZE": mean_squared_error(y_true_size, y_pred_df["Refined_SIZE"]),
        "MSE_PDI": mean_squared_error(y_true_pdi, y_pred_df["Refined_PDI"]),
        "MAE_SIZE": mean_absolute_error(y_true_size, y_pred_df["Refined_SIZE"]),
        "MAE_PDI": mean_absolute_error(y_true_pdi, y_pred_df["Refined_PDI"])
    }


# =============================
# 5. Greedy scaling
# =============================
def greedy_scale_search(X, y_size, y_pdi, scale_factors, cols_to_scale, model):
    X_scaled = X.copy()
    best_factors = {col: 1.0 for col in X.columns}

    for col in cols_to_scale:
        r2_per_factor = {}
        for factor in scale_factors:
            X_scaled[col] = X[col] * factor
            y_pred_df = model.predict(X_scaled)
            r2_size = r2_score(y_size, y_pred_df["Refined_SIZE"])
            r2_pdi = r2_score(y_pdi, y_pred_df["Refined_PDI"])
            r2_mean = (r2_size + r2_pdi) / 2
            r2_per_factor[factor] = r2_mean

        best_factor = max(r2_per_factor, key=r2_per_factor.get)
        best_factors[col] = best_factor
        X_scaled[col] = X[col] * best_factor
        print(f"[+] Feature {col}: miglior fattore {best_factor} -> R2 medio {r2_per_factor[best_factor]:.4f}")

    final_pred_df = model.predict(X_scaled)
    final_metrics = compute_metrics(y_size, y_pdi, final_pred_df)

    return best_factors, final_metrics


# =============================
# 6. Esecuzione
# =============================

# Step 1: baseline (solo RF = 0 epoche)
print("\n📊 Metriche iniziali (RF baseline, 0 epoche):")
baseline_model = RefinedModel(rf_model, xgb_size, xgb_pdi, num_epochs=0)
baseline_pred = baseline_model.predict(X_val)
baseline_metrics = compute_metrics(y_val_size, y_val_pdi, baseline_pred)
for k, v in baseline_metrics.items():
    print(f"  {k}: {v:.4f}")

# Step 2: iterative refinement (senza scaling)
print("\n📊 Metriche dopo iterative refinement (5 epoche, senza scaling):")
refined_pred = refined_model.predict(X_val)
refined_metrics = compute_metrics(y_val_size, y_val_pdi, refined_pred)
for k, v in refined_metrics.items():
    print(f"  {k}: {v:.4f}")

# Step 3: greedy scaling
scale_factors = np.round(np.arange(0.1, 20.01, 0.1), 2)
best_factors, best_metrics = greedy_scale_search(X_val, y_val_size, y_val_pdi, scale_factors, cols_to_scale, refined_model)

print("\nMigliori fattori di scala trovati (AQUEOUS esclusa):")
for col, f in best_factors.items():
    print(f"  {col}: {f}")

print("\n📊 Metriche finali con greedy scaling:")
for k, v in best_metrics.items():
    print(f"  {k}: {v:.4f}")
