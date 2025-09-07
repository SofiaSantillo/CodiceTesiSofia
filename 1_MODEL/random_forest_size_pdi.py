import pandas as pd
import pickle
import os
import sys
import logging
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np


######################################################################################################
# 1. Load Files and Configurations
######################################################################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from _File.config import logs_path, data_path, models_path, setup_logging

######################################################################################################
# 2. Modeling
######################################################################################################
def random_forest_size_pdi(DATA):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    setup_logging(logs_path, "random_forest_size_pdi.log")
    logger_RF=logging.getLogger("random_forest_size_pdi.log")
    
    logger_RF.info("Starting the Random Forest model training and evaluation process...")
    features = DATA.drop(columns=["SIZE", "PDI", "ID"])
    targets = DATA[["SIZE", "PDI"]]
    categorical_columns = features.select_dtypes(include=["object"]).columns
    numerical_columns = features.select_dtypes(include=["float64", "int64"]).columns
    # Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown='ignore'), categorical_columns)], 
        remainder="passthrough"
    )
    # Model Pipeline with Cross-Validation
    model = Pipeline(steps=[
        ("preprocessor", preprocessor), 
        ("regressor", MultiOutputRegressor(RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)))
    ])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2_scores = cross_val_score(model, features, targets, cv=kf, scoring='r2')
    cv_mse_scores = cross_val_score(model, features, targets, cv=kf, scoring='neg_mean_squared_error')
    cv_mae_scores = cross_val_score(model, features, targets, cv=kf, scoring='neg_mean_absolute_error')
    logger_RF.info("Cross-validation results:")
    logger_RF.info(f"Average R-squared: {np.mean(cv_r2_scores):.4f}")
    logger_RF.info(f"Average Mean Squared Error: {-np.mean(cv_mse_scores):.4f}")
    logger_RF.info(f"Average Mean Absolute Error: {-np.mean(cv_mae_scores):.4f}")
    # Train Final Model on Full Training Set
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)


    # Predict on Test Set
    y_pred = model.predict(X_test)

    y_test_size = y_test['SIZE']
    y_test_pdi = y_test['PDI']

    y_pred_size = y_pred [:, 0]
    y_pred_pdi = y_pred [:, 1]

    r2_size = r2_score(y_test_size, y_pred_size) 
    r2_pdi = r2_score(y_test_pdi, y_pred_pdi)

    mse_size = mean_squared_error(y_test_size, y_pred_size)
    mse_pdi = mean_squared_error(y_test_pdi, y_pred_pdi)

    mae_size = mean_absolute_error(y_test_size, y_pred_size)
    mae_pdi = mean_absolute_error(y_test_pdi, y_pred_pdi)

    logger_RF.info("Final model evaluation:")
    logger_RF.info(f"R-squared size: {r2_size:.4f}")
    logger_RF.info(f"R-squared pdi: {r2_pdi:.4f}")
    logger_RF.info(f"Mean Squared Error Size: {mse_size:.4f}")
    logger_RF.info(f"Mean Squared Error Pdi: {mse_pdi:.4f}")
    logger_RF.info(f"Normalized Mean Squared Error Size: {mse_size/np.var(y_test_size):.4f}")
    logger_RF.info(f"Normalized Mean Squared Error Pdi: {mse_pdi/np.var(y_test_pdi):.4f}")
    logger_RF.info(f"Mean Absolute Error Size: {mae_size:.4f}")
    logger_RF.info(f"Mean Absolute Error Pdi: {mae_pdi:.4f}")


    # Save Model
    model_path = os.path.join("1_MODEL/random_forest_size_pdi.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger_RF.info(f"Model saved successfully at {model_path}")

    
    ######################################################################################################
    # 3. Validation
    ######################################################################################################

    new_file_path = data_path + "/validation_1.csv"
    logger_RF.info(f"---> Loading validation dataset from {new_file_path}")
    VALIDATION_DATA = pd.read_csv(new_file_path)
    validation_features = VALIDATION_DATA.drop(columns=["SIZE", "PDI", "ID"], errors="ignore")
    validation_targets = VALIDATION_DATA[["SIZE", "PDI"]]
    # Predict on Validation Data
    validation_predictions = model.predict(validation_features)

    validation_targets_size = VALIDATION_DATA["SIZE"]
    validation_targets_pdi= VALIDATION_DATA["PDI"]

    validation_predictions_size = validation_predictions [ :, 0]
    validation_predictions_pdi = validation_predictions [ :, 1]
    # Evaluate Validation Performance
    validation_r2_size = r2_score(validation_targets_size, validation_predictions_size)
    validation_r2_pdi = r2_score(validation_targets_pdi, validation_predictions_pdi)
    
    validation_mse_size = mean_squared_error(validation_targets_size, validation_predictions_size)
    validation_mse_pdi = mean_squared_error(validation_targets_pdi, validation_predictions_pdi)

    validation_mae_size = mean_absolute_error(validation_targets_size, validation_predictions_size)
    validation_mae_pdi = mean_absolute_error(validation_targets_pdi, validation_predictions_pdi)

    logger_RF.info("Model validation completed successfully.")
    logger_RF.info(f"Validation R-squared Size: {validation_r2_size:.4f}")
    logger_RF.info(f"Validation R-squared Pdi: {validation_r2_pdi:.4f}")

    logger_RF.info(f"Validation Mean Squared Error Size: {validation_mse_size:.4f}")
    logger_RF.info(f"Validation Mean Squared Error Pdi: {validation_mse_pdi:.4f}")

    logger_RF.info(f"Validation Normalized Mean Squared Error Size: {validation_mse_size/np.var(validation_targets_size):.4f}")
    logger_RF.info(f"Validation Normalized Mean Squared Error Pdi: {validation_mse_pdi/np.var(y_test_pdi):.4f}")

    logger_RF.info(f"Validation Mean Absolute Error Size: {validation_mae_size:.4f}")
    logger_RF.info(f"Validation Mean Absolute Error Pdi: {validation_mae_pdi:.4f}")

    logger_RF.info("...DONE!\n\n")

    DATA[["SIZE", "PDI"]] = model.predict(features)

    return validation_r2_pdi, validation_r2_size, DATA, model
    
    


#if __name__== "__main__":
 #   file_path = data_path + "/seed.csv"
  #  DATA = pd.read_csv(file_path)
   # DATA.dropna(inplace=True)

    #sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    #from _File.config import data_path

    #validation_r2_pdi, validation_r2_size, DATA, model = random_forest_size_pdi(DATA)

