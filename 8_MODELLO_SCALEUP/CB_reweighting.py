import sys
import numpy as np
import pandas as pd
import os
import logging
import pickle
from joblib import load
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from _File.config import data_path, models_path, setup_logging

logs_path="8_MODELLO_SCALEUP/_log"
setup_logging(logs_path, "Model_predictive_cb_complete.log")
logging.info("\n\n\nStarting importance sampling correction on validation set...")

# --- Load datasets ---
file_path = data_path + "/data_1.csv"
validation_file_path = data_path + "/dataset_ScaleUp_mixed_validation.csv"
DATA = pd.read_csv(file_path).dropna()
validation_data = pd.read_csv(validation_file_path).dropna()

X_validation = validation_data.drop(columns=["SIZE", "PDI"])
y_val = validation_data[["SIZE", "PDI"]]

# --- Load pre-trained models ---
rf_model = load(models_path + '/best_random_forest_size_pdi_data_1.pkl')
xgb_size = load(models_path + '/best_xgboost_model__data_1_size.pkl')
xgb_pdi = load(models_path + '/best_xgboost_model__data_1_pdi.pkl')

# --- Iterative refinement function ---
def iterative_refinement(X_raw, num_epochs=2):
    X = X_raw.copy()
    initial_preds = rf_model.predict(X)
    size_pred = initial_preds[:, 0].ravel() 
    pdi_pred = initial_preds[:, 1].ravel() 
    for epoch in range(num_epochs):
        X_with_pdi = X.copy()
        X_with_pdi['PDI'] = pdi_pred
        size_pred = xgb_size.predict(X_with_pdi).ravel() 
        X_with_size = X.copy()
        X_with_size['SIZE'] = size_pred
        pdi_pred = xgb_pdi.predict(X_with_size).ravel()  
    return np.vstack([size_pred, pdi_pred]).T  # shape=(N_samples, 2)

# --- Causal Bootstrap weights for multivariate target ---
def compute_cb_weights_multivariate(df, targets, U, Z, D=None):
    N = len(df)
    weights = np.ones(N)
    for n in range(N):
        row = df.iloc[n]
        w_n = 1.0
        # Confounders
        for u in U:
            counts = df[u].value_counts(normalize=True)
            w_n *= counts.get(row[u], 1.0/N)
        # Mediators
        for z in Z:
            mask = np.ones(len(df), dtype=bool)
            for t in targets:
                mask &= df[t] == row[t]
            counts = df[mask][z].value_counts(normalize=True)
            w_n *= counts.get(row[z], 1.0/N)
        # Decision D (optional)
        if D:
            for d in D:
                subset = df.copy()
                for t in targets:
                    subset = subset[subset[t] == row[t]]
                for u in U:
                    subset = subset[subset[u] == row[u]]
                counts = subset[d].value_counts(normalize=True)
                w_n *= counts.get(row[d], 1.0/N)
        weights[n] = w_n
    weights /= np.sum(weights)
    return weights

# --- Apply iterative refinement to validation set ---
y_hat = iterative_refinement(X_validation, num_epochs=5)

# --- Define U, Z, D as already identified ---
U = ['ESM', 'PEG']  # union of confounders SIZE+PDI
Z = ['ESM', 'PDI', 'CHOL', 'PEG', 'SIZE', 'AQUEOUS', 'HSPC']  # union of mediators SIZE+PDI
D = []  # no decision variables

# --- Compute CB weights ---
weights = compute_cb_weights_multivariate(validation_data, ['SIZE','PDI'], U, Z, D)

# --- Apply importance sampling (weighted predictions) ---
y_hat_cb_weighted = y_hat * weights[:, np.newaxis]
y_hat_cb_corr = y_hat_cb_weighted  # sample-wise corrected predictions

# --- Calculate metrics ---
# Original predictions
r2_size_orig = r2_score(y_val['SIZE'], y_hat[:,0])
r2_pdi_orig  = r2_score(y_val['PDI'], y_hat[:,1])
mse_size_orig = mean_squared_error(y_val['SIZE'], y_hat[:,0])
mse_pdi_orig  = mean_squared_error(y_val['PDI'], y_hat[:,1])
mae_size_orig = mean_absolute_error(y_val['SIZE'], y_hat[:,0])
mae_pdi_orig  = mean_absolute_error(y_val['PDI'], y_hat[:,1])

# CB-corrected predictions
r2_size_cb = r2_score(y_val['SIZE'], y_hat_cb_corr[:,0])
r2_pdi_cb  = r2_score(y_val['PDI'], y_hat_cb_corr[:,1])
mse_size_cb = mean_squared_error(y_val['SIZE'], y_hat_cb_corr[:,0])
mse_pdi_cb  = mean_squared_error(y_val['PDI'], y_hat_cb_corr[:,1])
mae_size_cb = mean_absolute_error(y_val['SIZE'], y_hat_cb_corr[:,0])
mae_pdi_cb  = mean_absolute_error(y_val['PDI'], y_hat_cb_corr[:,1])

logging.info(f"Original metrics:")
logging.info(f"SIZE -> R²: {r2_size_orig:.4f}, MSE: {mse_size_orig:.4f}, MAE: {mae_size_orig:.4f}")
logging.info(f"PDI  -> R²: {r2_pdi_orig:.4f}, MSE: {mse_pdi_orig:.4f}, MAE: {mae_pdi_orig:.4f}")
logging.info(f"CB-corrected metrics (sample-wise weighted):")
logging.info(f"SIZE -> R²: {r2_size_cb:.4f}, MSE: {mse_size_cb:.4f}, MAE: {mae_size_cb:.4f}")
logging.info(f"PDI  -> R²: {r2_pdi_cb:.4f}, MSE: {mse_pdi_cb:.4f}, MAE: {mae_pdi_cb:.4f}")

print("Original metrics:")
print(f"SIZE -> R²: {r2_size_orig:.4f}, MSE: {mse_size_orig:.4f}, MAE: {mae_size_orig:.4f}")
print(f"PDI  -> R²: {r2_pdi_orig:.4f}, MSE: {mse_pdi_orig:.4f}, MAE: {mae_pdi_orig:.4f}")
print("CB-corrected metrics (sample-wise weighted):")
print(f"SIZE -> R²: {r2_size_cb:.4f}, MSE: {mse_size_cb:.4f}, MAE: {mae_size_cb:.4f}")
print(f"PDI  -> R²: {r2_pdi_cb:.4f}, MSE: {mse_pdi_cb:.4f}, MAE: {mae_pdi_cb:.4f}")

# --- Save results ---
output_path = os.path.join(models_path, "refined_model_cb_complete.pkl")
with open(output_path, "wb") as f:
    pickle.dump({
        'rf_model': rf_model,
        'xgb_size': xgb_size,
        'xgb_pdi': xgb_pdi,
        'weights_cb': weights,
        'y_hat_refined': y_hat,
        'y_hat_cb_corrected': y_hat_cb_corr,
        'metrics_original': {
            'r2': (r2_size_orig, r2_pdi_orig),
            'mse': (mse_size_orig, mse_pdi_orig),
            'mae': (mae_size_orig, mae_pdi_orig)
        },
        'metrics_cb_corrected': {
            'r2': (r2_size_cb, r2_pdi_cb),
            'mse': (mse_size_cb, mse_pdi_cb),
            'mae': (mae_size_cb, mae_pdi_cb)
        }
    }, f)

logging.info(f"CB-corrected predictions and metrics saved at {output_path}")
