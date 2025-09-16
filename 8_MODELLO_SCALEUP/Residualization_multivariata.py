import sys
import os
import logging
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from joblib import load
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from _File.config import data_path, models_path, setup_logging

logs_path="8_MODELLO_SCALEUP/_log"
setup_logging(logs_path, "Model_residualization.log")
logging.info("Starting residualization correction on validation set...")

# --- Load datasets ---
file_path = data_path + "/data_1.csv"
validation_file_path = data_path + "/dataset_ScaleUp_mixed_train.csv"
DATA = pd.read_csv(file_path).dropna()
validation_data = pd.read_csv(validation_file_path).dropna()

X_validation = validation_data.drop(columns=["SIZE", "PDI"])
y_val = validation_data[["SIZE", "PDI"]]

# --- Load pre-trained models ---
rf_model = load(models_path + '/best_random_forest_size_pdi_data_1.pkl')
xgb_size = load(models_path + '/best_xgboost_model__data_1_size.pkl')
xgb_pdi = load(models_path + '/best_xgboost_model__data_1_pdi.pkl')

# --- Confounders U già identificati ---
U = ['ESM', 'PEG']  # unione dei confounders SIZE+PDI

# --- Step 1: Fit linear model Y ~ U sui dati di training ---
X_train_U = DATA[U]
y_train = DATA[["SIZE", "PDI"]]

linear_model = LinearRegression()
linear_model.fit(X_train_U, y_train)
logging.info("Linear regression for residualization trained successfully.")

# --- Step 2: Calcolo dei residui sui dati di validazione ---
X_val_U = validation_data[U]
y_pred_conf = linear_model.predict(X_val_U)
residuals = y_val.values - y_pred_conf  # Y_residual = Y - E[Y|U]

# --- Step 3: Iterative refinement sul validation set ---
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

y_hat = iterative_refinement(X_validation, num_epochs=5)

# --- Step 4: Residualization post-hoc ---
# Y_hat_corrected = Y_hat - E[Y|U]_val + E[Y|U]_train_mean
# Questo riporta la scala predittiva al range originale
y_hat_corrected = y_hat - linear_model.predict(X_val_U) + np.mean(y_train.values, axis=0)

# --- Step 5: Metriche su validation set ---
r2_size = r2_score(y_val["SIZE"], y_hat_corrected[:,0])
r2_pdi  = r2_score(y_val["PDI"],  y_hat_corrected[:,1])
mse_size = mean_squared_error(y_val["SIZE"], y_hat_corrected[:,0])
mse_pdi  = mean_squared_error(y_val["PDI"],  y_hat_corrected[:,1])
mae_size = mean_absolute_error(y_val["SIZE"], y_hat_corrected[:,0])
mae_pdi  = mean_absolute_error(y_val["PDI"], y_hat_corrected[:,1])

logging.info("Residualization-corrected predictions metrics:")
logging.info(f"SIZE -> R²: {r2_size:.4f}, MSE: {mse_size:.4f}, MAE: {mae_size:.4f}")
logging.info(f"PDI  -> R²: {r2_pdi:.4f}, MSE: {mse_pdi:.4f}, MAE: {mae_pdi:.4f}")

# --- Step 6: Save results ---
output_path = os.path.join(models_path, "refined_model_residualized.pkl")
with open(output_path, "wb") as f:
    pickle.dump({
        'rf_model': rf_model,
        'xgb_size': xgb_size,
        'xgb_pdi': xgb_pdi,
        'linear_model_U': linear_model,
        'y_hat_corrected': y_hat_corrected,
        'y_val': y_val
    }, f)

logging.info(f"Residualization-corrected predictions saved at {output_path}")
