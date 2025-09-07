import os
import sys
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
from xgboost_gridsearch_pdi import functionalize_xgboost_gridsearch_pdi
from xgboost_gridsearch_size import functionalize_xgboost_gridsearch_size
from random_forest_size_pdi import random_forest_size_pdi

import joblib




if __name__=="__main__":
    max_iter=10000
    threshold_size=0.85
    threshold_pdi=0.75

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from _File.config import data_path

    file_path = data_path + "/seed_non_ordinato.csv"
    DATA = pd.read_csv(file_path)
    DATA.dropna(inplace=True)

    r2_size_history = []
    r2_pdi_history = []
    
    rf_r2_pdi, rf_r2_size, data, model_rf= random_forest_size_pdi(DATA=DATA)
    best_rf_r2_size = rf_r2_size
    best_rf_r2_pdi = rf_r2_pdi

    r2_size_history.append(best_rf_r2_size)
    r2_pdi_history.append(best_rf_r2_pdi)
      

    for i in range(max_iter):
        if best_rf_r2_size >= threshold_size and best_rf_r2_pdi >= threshold_pdi:
            print(f"Early stopping at iteration {i+1} due to insufficient improvement.")
            break
        if(best_rf_r2_pdi > best_rf_r2_size): 
            data = functionalize_xgboost_gridsearch_pdi(data=data)
            data = functionalize_xgboost_gridsearch_size(data=data)
        else:   
            data = functionalize_xgboost_gridsearch_size(data=data)
            data = functionalize_xgboost_gridsearch_pdi(data=data)

        rf_r2_pdi, rf_r2_size, data, model_rf= random_forest_size_pdi(data)
            
        best_rf_r2_pdi = rf_r2_pdi
        best_rf_r2_size = rf_r2_size

        r2_size_history.append(best_rf_r2_size)
        r2_pdi_history.append(best_rf_r2_pdi)
 


    plt.figure(figsize=(10, 5))
    plt.plot(r2_size_history, label='R² SIZE', marker='o')
    plt.plot(r2_pdi_history, label='R² PDI', marker='x')
    plt.xlabel('Iterazione')
    plt.ylabel('R²')
    plt.title('Evoluzione di R² per SIZE e PDI')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('_Plot/r2_evolution.png') 
    plt.show()


    # Salva il modello Random Forest addestrato
    filename_model = "6_ANALISI_INTERVENTISTA/model_rf.pkl"  # percorso dove vuoi salvare il modello
    os.makedirs(os.path.dirname(filename_model), exist_ok=True)  # crea cartella se non esiste
    joblib.dump(model_rf, filename_model)

    print(f"Modello salvato in {filename_model}")


