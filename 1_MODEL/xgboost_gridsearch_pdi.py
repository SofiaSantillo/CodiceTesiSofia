import pandas as pd
import pickle
import os
import sys
import logging
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from random_forest_size_pdi import random_forest_size_pdi

######################################################################################################
# 1. Load Files and Configurations
######################################################################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from _File.config import * #importo tutto quello che c'è in config



#######################################################################################################
# 2. Functionalization of xgboost with different seed and optimal param for grid search
#######################################################################################################

def functionalize_xgboost_gridsearch_pdi(data):
    """function that allows me to run xgboost with different seeds and with grid search with optimal parameters"""

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    setup_logging(logs_path, "xgboost_gridsearch_pdi.log")

    file_path = data_path + "/seed_non_ordinato.csv" 

    logger_PDI = logging.getLogger("xgboost_gridsearch_pdi.log")

    logger_PDI.info("Starting the XGBoost model training process for predicting PDI...".upper())
    logger_PDI.info(f"---> Loading dataset from {file_path}")

    features = data.drop(columns=["PDI", "ID"])
    targets = data[["PDI"]]
    categorical_columns = features.select_dtypes(include=["object"]).columns
    numerical_columns = features.select_dtypes(include=["float64", "int64"]).columns
    # Define the preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown='ignore'), categorical_columns)], 
        remainder="passthrough"
    )

    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

    xgbr = XGBRegressor(random_state=42)
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", xgbr)
    ])
    
    param_grid = {
        "regressor__n_estimators": [50, 100, 200],
        "regressor__max_depth": [3, 5, 7],
        "regressor__learning_rate": [0.01, 0.05, 0.1],
        "regressor__subsample": [0.6, 0.8, 1.0],
        "regressor__colsample_bytree": [0.6, 0.8, 1.0],
        "regressor__gamma": [0, 1, 5]
    }
    # Grid Search
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1) 
    grid_search.fit(X_train, y_train) 
    # Predict and evaluate on test data
    y_pred_full = grid_search.predict(features)
    y_pred= grid_search.predict(X_test)
    logger_PDI.info(f"Best Parameters: {grid_search.best_params_}")
    
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    logger_PDI.info("Model training completed successfully.")
    logger_PDI.info(f"R-squared_SIZE: {r2}")
    logger_PDI.info(f"Mean Squared Error_SIZE: {mse}")  
    logger_PDI.info(f"Mean Absolute Error_SIZE: {mae}")


    # Save the model
    model_path = os.path.join("1_MODEL/xgboost_gridsearch_pdi.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(grid_search.best_estimator_, f)
    logger_PDI.info(f"Model saved successfully at {model_path}")

    #######################################################################################################
    # 3. Validation
    #######################################################################################################
    new_file_path = data_path + "/validation_1.csv" 
    logger_PDI.info(f"---> Loading validation dataset from {new_file_path}")
    # Load validation data 
    VALIDATION_DATA = pd.read_csv(new_file_path)
    validation_features = VALIDATION_DATA.drop(columns=["PDI", "ID"], errors="ignore") 
    validation_targets = VALIDATION_DATA[["PDI"]]
    # Load the trained model
    with open(model_path, "rb") as f:
        loaded_model = pickle.load(f)
    # Predict and evaluate
    validation_predictions = loaded_model.predict(validation_features) 

    validation_r2 = r2_score(validation_targets, validation_predictions)
    validation_mse = mean_squared_error(validation_targets, validation_predictions)
    validation_mae = mean_absolute_error(validation_targets, validation_predictions)
    logger_PDI.info("Model validation completed successfully.")
    logger_PDI.info(f"Validation R-squared_SIZE: {validation_r2}")
    logger_PDI.info(f"Validation Mean Squared Error_SIZE: {validation_mse}")
    logger_PDI.info(f"Validation Mean Absolute Error_SIZE: {validation_mae}")
    logger_PDI.info("...VALIDATION DONE!\n\n")


    data['PDI']=y_pred_full
    return data


#if __name__ == "__main__":
 
 #   file_path = data_path + "/seed_non_ordinato.csv"
  #  DATA = pd.read_csv(file_path)
   # DATA.dropna(inplace=True)
    
    #rf_r2_pdi, rf_r2_size, data= random_forest_size_pdi(DATA)

    #data= functionalize_xgboost_gridsearch_pdi(data=data)
    #logging.info("...Done!\n\n".upper())