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

sys.stdout = Logger("8_MODELLO_SCALEUP/_log/Model_scalarization_greedSearch_range_weights(mean_mse).log")

print("\n-------------  START MODEL SCALARIZATION GREED SEARCH (scale factors + weights) ------------------")

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
        
        best_mse = np.inf
        best_factor = 1.0
        
        for f in factors:
            temp_scaled = X_col.copy()
            temp_scaled[mask] = X_col[mask] * f
            X_temp = X_val.copy()
            X_temp[X_col.name] = temp_scaled
            
            y_pred_df = iterative_refinement_predict(X_temp, rf_model, xgb_size, xgb_pdi, num_epochs=num_epochs)
            mse_mean = (mean_squared_error(y_size, y_pred_df["Refined_SIZE"]) +
                        mean_squared_error(y_pdi, y_pred_df["Refined_PDI"])) / 2
            
            if mse_mean < best_mse:
                best_mse = mse_mean
                best_factor = f
        
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
            factors = scale_ranges[col]
            mse_per_factor = {}
            for f in factors:
                X_scaled[col] = X[col] * f
                y_pred_df = iterative_refinement_predict(X_scaled, rf_model, xgb_size, xgb_pdi, num_epochs=5)
                mse_mean = (mean_squared_error(y_size, y_pred_df["Refined_SIZE"]) +
                            mean_squared_error(y_pdi, y_pred_df["Refined_PDI"])) / 2
                mse_per_factor[f] = mse_mean
            best_factor = min(mse_per_factor, key=mse_per_factor.get)
            best_factors[col] = best_factor
            X_scaled[col] = X[col] * best_factor
            print(f"[+] Feature {col}: miglior fattore {best_factor} -> MSE medio {mse_per_factor[best_factor]:.4f}")
    
    final_pred_df = iterative_refinement_predict(X_scaled, rf_model, xgb_size, xgb_pdi, num_epochs=5)
    final_metrics = compute_metrics(y_size, y_pdi, final_pred_df)
    
    return best_factors, final_metrics

def optimize_global_weight(col_scaled, X_full, y_size, y_pdi, rf_model, xgb_size, xgb_pdi,
                           num_epochs=5, weight_range=(0.1,3.0), step=0.01):
    best_weight = 1.0
    best_mse = np.inf
    weights = np.arange(weight_range[0], weight_range[1]+step, step)
    
    for w in weights:
        X_temp = X_full.copy()
        X_temp[col_scaled.name] = col_scaled * w
        y_pred_df = iterative_refinement_predict(X_temp, rf_model, xgb_size, xgb_pdi, num_epochs=num_epochs)
        mse_mean = (mean_squared_error(y_size, y_pred_df["Refined_SIZE"]) +
                    mean_squared_error(y_pdi, y_pred_df["Refined_PDI"])) / 2
        if mse_mean < best_mse:
            best_mse = mse_mean
            best_weight = w
    
    col_final = col_scaled * best_weight
    return col_final, best_weight, best_mse

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

X_scaled_final = X_val.copy()

for col in cols_to_scale:
    if isinstance(scale_ranges[col], dict):
        scaled_col, _ = apply_cluster_scale(
            X_scaled_final[col], scale_ranges[col], y_val_size, y_val_pdi,
            rf_model, xgb_size, xgb_pdi, num_epochs=5
        )
        X_scaled_final[col] = scaled_col
    else:
        X_scaled_final[col] = X_val[col] * best_factors[col]

y_pred_final = iterative_refinement_predict(X_scaled_final, rf_model, xgb_size, xgb_pdi, num_epochs=5)
X_scaled_final["SIZE"] = y_pred_final["Refined_SIZE"]
X_scaled_final["PDI"] = y_pred_final["Refined_PDI"]

feature_weight= {"TFR", "PEG", "ESM"}
for f in feature_weight:
    print(f"\n🔹 Ottimizzazione peso globale per {f} dopo cluster-specific scaling...")
    scaled_col = X_scaled_final[f].copy()
    scaled_col_final, best_weight, best_mse = optimize_global_weight(
        scaled_col, X_scaled_final, y_val_size, y_val_pdi,
        rf_model, xgb_size, xgb_pdi, num_epochs=5,
        weight_range=(0.1,3.0), step=0.01
    )
    X_scaled_final[f] = scaled_col_final
    print(f"Peso ottimale {f}: {best_weight:.3f} -> MSE medio: {best_mse:.4f}")

y_pred_final = iterative_refinement_predict(X_scaled_final, rf_model, xgb_size, xgb_pdi, num_epochs=5)
X_scaled_final["SIZE"] = y_pred_final["Refined_SIZE"]
X_scaled_final["PDI"] = y_pred_final["Refined_PDI"]

final_metrics = compute_metrics(y_val_size, y_val_pdi, y_pred_final)

print("\n📊 Metriche finali sulle predizioni con pesi globali ottimizzati:")
for k, v in final_metrics.items():
    print(f"  {k}: {v:.4f}")
