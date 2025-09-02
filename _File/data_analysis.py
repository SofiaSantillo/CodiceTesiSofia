import pandas as pd
import matplotlib.pyplot as plt
import logging
import io
import os
import sys
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import umap
import plotly.graph_objects as go


######################################################################################################
# 1. Load Files and Configurations
######################################################################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from _File.config import *


setup_logging(logs_path, "exploratory_data_analysis.log")



DATA_PATH = data_path
PLOT_DIR = "_Plot/Data_Analysis"
seed_filename= "seed.csv"
csv_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.csv')]
data = pd.read_csv(os.path.join(DATA_PATH, seed_filename))



######################################################################################################
# 2. Processing and Saving  Functions
######################################################################################################
def save_plot(fig, plot_name):
    """Save plots with proper filename handling."""
    fig.savefig(f"{PLOT_DIR}/{plot_name}.png")
    plt.close()
    logging.info("---> Plot saved at: %s", PLOT_DIR)


def standardize_data(data):
    """Standardize the data for PCA and UMAP."""
    scaler = StandardScaler()
    return scaler.fit_transform(data)


def modify_categorical_columns(data):
    """Sostituire il primo elemento di ogni colonna categorica con 1 e il secondo con 0."""
    # Seleziona le colonne categoriche
    categorical_data = data.select_dtypes(exclude=['number']).copy()

    # Itera su ogni colonna categorica
    for col in categorical_data.columns:
        # Prendi i valori unici nella colonna
        unique_values = categorical_data[col].unique()
        
        # Verifica se la colonna ha almeno due valori unici
        if len(unique_values) >= 2:
            # Sostituisci il primo valore unico con 1 e il secondo con 0
            first_value = unique_values[0]
            second_value = unique_values[1]
            
            # Applica la modifica nella colonna
            categorical_data[col] = categorical_data[col].replace({first_value: 1, second_value: 0})
    
    return categorical_data



def correlation_heatmap(data, filename):
    """Get correlation heatmap."""
    numeric_data = data.select_dtypes(include=['number']).dropna()

    categorical_data = data.select_dtypes(exclude=['number']).dropna()
    # Esempio di utilizzo:
    modified_data = modify_categorical_columns(categorical_data)

    # Combina dati numerici e dati categoriali trasformati
    full_data = pd.concat([numeric_data, modified_data], axis=1)
    
    # Calcola la matrice di correlazione
    corr_matrix = full_data.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    plt.title(f"Feature Correlation Heatmap for {filename}")
    save_plot(fig, f"{filename}_correlation_heatmap")
    logging.info("---> Correlation heatmap at: %s", f"{PLOT_DIR}/{filename}_correlation_heatmap.png")


def feature_importance(data, target_features=["SIZE", "PDI"], filename="feature_importance"):
    """Train a Random Forest model and save feature importance plots."""
    logging.info(f"Calculating feature importance for {filename}...")

    numeric_data = data.select_dtypes(include=['number'])
    feature_columns = [col for col in numeric_data.columns if col not in target_features]

    for target_feature in target_features:
        # Rimuovi righe con NaN in X o y
        df_temp = numeric_data[feature_columns + [target_feature]].dropna()
        X = df_temp[feature_columns]
        y = df_temp[target_feature]

        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)

        feature_importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        fig = plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importances,
            palette='viridis', hue='Feature', legend=False)
        plt.title(f"Feature Importance for {target_feature} in {filename}")

        save_plot(fig, f"{filename}_feature_importance_{target_feature}")
        logging.info("---> Feature importance at: %s", f"{PLOT_DIR}/{filename}_feature_importance_{target_feature}.png")




######################################################################################################
# 3. Pipeline
######################################################################################################
def run_pipeline():
    """Main function to execute the exploratory data analysis pipeline."""
    logging.info("Starting Exploratory Data Analysis (EDA)...".upper())
    
    correlation_heatmap(data=data, filename=seed_filename)
 
    feature_importance(data=data, filename=seed_filename)
   


if __name__ == "__main__":
    run_pipeline()
    logging.info("...Done!\n\n".upper())