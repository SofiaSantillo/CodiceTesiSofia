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

sys.stdout = Logger("8_MODELLO_SCALEUP/_log/Model_scalarization_greedSearch_range(vincolo mse).log")

print( "\n-------------  START MODEL SCALARIZATION GREED SEARCH (scale factors) - mse ------------------")

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

def iterative_refinement_predict(X_raw, rf_model, xgb_size, xgb_pdi, num_epochs=5):
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

def apply_cluster_scale(X_col, cluster_factors, y_size, y_pdi, rf_model, xgb_size, xgb_pdi, num_epochs=5):
    scaled_col = X_col.copy()
    best_factors = {}

    for cluster, factors in cluster_factors.items():
        low, high = map(float, cluster.split('-'))
        mask = (X_col >= low) & (X_col <= high)

        valid_candidates = []
        all_candidates = []
        for f in factors:
            temp_scaled = X_col.copy()
            temp_scaled[mask] = X_col[mask] * f
            X_temp = X_val.copy()
            X_temp[X_col.name] = temp_scaled

            y_pred_df = iterative_refinement_predict(X_temp, rf_model, xgb_size, xgb_pdi, num_epochs=num_epochs)
            mse_size = mean_squared_error(y_size, y_pred_df["Refined_SIZE"])
            mse_pdi = mean_squared_error(y_pdi, y_pred_df["Refined_PDI"])
            all_candidates.append((f, mse_size, mse_pdi))

            if mse_size <= 80:
                valid_candidates.append((f, mse_size, mse_pdi))

        if valid_candidates:
            best_factor, _, _ = min(valid_candidates, key=lambda x: x[1] + x[2])
        else:
            best_factor, _, _ = min(all_candidates, key=lambda x: x[1] + x[2])
            print(f"[!] Nessun fattore valido per cluster {cluster}, scelto il migliore trovato: {best_factor}")

        scaled_col[mask] = X_col[mask] * best_factor
        best_factors[cluster] = best_factor

    return scaled_col, best_factors

def greedy_scale_search(X, y_size, y_pdi, scale_ranges, cols_to_scale):
    X_scaled = X.copy()
    best_factors = {}

    for col in cols_to_scale:
        if col not in scale_ranges:
            print(f"[!] Nessun range specificato per {col}, uso default [0.1, 20.0]")
            factors = np.round(np.arange(0.1, 20.01, 0.1), 2)
            scale_ranges[col] = factors

        if isinstance(scale_ranges[col], dict):
            X_scaled[col], best_factors[col] = apply_cluster_scale(
                X_scaled[col], scale_ranges[col], y_size, y_pdi, rf_model, xgb_size, xgb_pdi, num_epochs=5
            )
            print(f"[+] Feature {col}: applicato cluster-specific scaling -> {best_factors[col]}")
        else:
            valid_candidates = []
            all_candidates = []
            for f in scale_ranges[col]:
                X_scaled[col] = X[col] * f
                y_pred_df = iterative_refinement_predict(X_scaled, rf_model, xgb_size, xgb_pdi, num_epochs=5)
                mse_size = mean_squared_error(y_size, y_pred_df["Refined_SIZE"])
                mse_pdi = mean_squared_error(y_pdi, y_pred_df["Refined_PDI"])
                all_candidates.append((f, mse_size, mse_pdi))

                if mse_size <= 80 :
                    valid_candidates.append((f, mse_size, mse_pdi))

            if valid_candidates:
                best_factor, _, _ = min(valid_candidates, key=lambda x: x[1] + x[2])
                print(f"[+] Feature {col}: trovato fattore valido {best_factor}")
            else:
                best_factor, _, _ = min(all_candidates, key=lambda x: x[1] + x[2])
                print(f"[!] Nessun fattore valido per {col}, scelto il migliore trovato: {best_factor}")

            best_factors[col] = best_factor
            X_scaled[col] = X[col] * best_factor

    final_pred_df = iterative_refinement_predict(X_scaled, rf_model, xgb_size, xgb_pdi, num_epochs=5)
    final_metrics = compute_metrics(y_size, y_pdi, final_pred_df)

    return best_factors, final_metrics


scale_ranges = {
    "TFR": {
        "0-3": np.round(np.arange(0.101, 1.2, 0.01), 2),   
        "3-25": np.round(np.arange(0.01, 0.97, 0.01), 3),
        "25-55": np.round(np.arange(0.001, 0.1, 0.001), 3)
    },
    "PEG": {
        "0-3": np.round(np.arange(0.01, 20.2, 0.1), 2),   
        "3-6": np.round(np.arange(0.01, 0.97, 0.01), 3)
    },
    "ESM": {
        "0-3": np.round(np.arange(0.01, 20.1, 0.1), 2),    
        "15-25": np.round(np.arange(0.60, 1.1, 0.01), 2)   
    },
    "FRR": {
        "0-5": np.round(np.arange(0.1, 3.1, 0.01), 2),    
        "5-10": np.round(np.arange(0.80, 1.8, 0.01), 2)   
    },
    "CHOL": {
        "0-5": np.round(np.arange(0.01, 1.3, 0.01), 2),    
        "5-15": np.round(np.arange(0.70, 1.3, 0.01), 2)   
    },
    "HSPC": {
        "0-5": np.round(np.arange(0.01, 0.5, 0.01), 2),    
        "5-10": np.round(np.arange(0.60, 1.3, 0.01), 2),
        "10-15": np.round(np.arange(0.80, 1.7, 0.01), 2),
        "15-25": np.round(np.arange(0.90, 2.3, 0.01), 2)   
    }
}


print("\n📊 Metriche dopo iterative refinement (senza scaling):")
refined_pred = iterative_refinement_predict(X_val, rf_model, xgb_size, xgb_pdi, num_epochs=5)
refined_metrics = compute_metrics(y_val_size, y_val_pdi, refined_pred)
for k, v in refined_metrics.items():
    print(f"  {k}: {v:.4f}")

best_factors, best_metrics = greedy_scale_search(X_val, y_val_size, y_val_pdi, scale_ranges, cols_to_scale)

print("\nMigliori fattori di scala trovati (AQUEOUS esclusa):")
for col, f in best_factors.items():
    print(f"  {col}: {f}")

print("\n📊 Metriche finali con greedy scaling:")
for k, v in best_metrics.items():
    print(f"  {k}: {v:.4f}")

print("\nMigliori fattori di scala trovati (AQUEOUS esclusa):")
for col, f in best_factors.items():
    print(f"  {col}: {f}")

print("\n📊 Metriche finali con greedy scaling:")
for k, v in best_metrics.items():
    print(f"  {k}: {v:.4f}")
