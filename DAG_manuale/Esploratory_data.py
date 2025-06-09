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
import numpy as np
import pickle
from scipy.stats import gaussian_kde

def setup_logging(root, path):
    """Set up logging configuration."""
    logging.basicConfig(
        filename= root + "/" + path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

LOG_PATH="_Logs"
setup_logging(LOG_PATH, "exploratory_data_analysis.log")

PLOT_DIR="_Plot"
DATA_PATH = "Data_DAG"
csv_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.csv')]

data_dict = {}
for file in csv_files:
    file_path = os.path.join(DATA_PATH, file)
    data_dict[file] = pd.read_csv(file_path)


def save_plot(fig, plot_name):
    logging.info("(----------------SAVE PLOT------------------)")
    """Save plots with proper filename handling.
    Args:
        fig(figura): la figura da salvare
        plot_name(file): file in cui salvare la figura
    Returns: """
    fig.savefig(f"{PLOT_DIR}/{plot_name}.png") 
    plt.close() 
    logging.info("---> Plot saved at: %s", PLOT_DIR) 



def data_explorer(data, filename, DAG):
    logging.info("(----------------DATA EXPLORER------------------)")
    """Perform exploratory data analysis (EDA) on the given dataset.
    Args:
        data(DataFrame): matrice da analizzare
        filename(string): nome del file in cui salvare il grafico che creeremo
    Returns: la funzione non ritorna nulla, semplicemente crea un grafico con degli istogrammi di cui calcola le varie statistiche e le salva su un log"""
    logging.info(f"Exploring data in {filename}")
    buffer = io.StringIO() 
    data.info(buf=buffer) 
    logging.info(f"Dataset Info for {filename}:\n{buffer.getvalue()}") 
    logging.info(f"Missing Values for {filename}:\n{data.isnull().sum().to_string()}") 
    logging.info(f"Summary Statistics for {filename}:\n{data.describe().to_string()}") 
    logging.info(f"First 5 Rows for {filename}:\n{data.head().to_string()}") 
    data.hist(bins=30, figsize=(12, 10)) 
    plt.suptitle(f"Feature Distributions for {filename}") 
    save_plot(plt, f"{DAG}_complete_data_histogram") 
    logging.info("---> Feature distributions at: %s", f"{PLOT_DIR}/{DAG}_complete_data_histogram.png")

def plot_numeric_distributions(dataset, filename, DAG):
    logging.info("(----------------PLOT NUMERIC DISTRIBUTION------------------)")
    """Plot and save the KDE distributions for all numeric columns in the dataset
    Args:
        dataset (DataFrame): dataset da analizzare
        filename (string): nome del file per il salvataggio del plot
    Returns: """
    logging.info(f"Plotting distributions for {filename}")
    numeric_data = dataset.select_dtypes(include=['number']).dropna()  # Seleziona tutte le variabili numeriche
    for column in numeric_data.columns:
        plt.figure(figsize=(6, 4))
        sns.kdeplot(numeric_data[column], label=f"{column} Distribution", fill=True, color="skyblue")
        plt.title(f"Distribution of {column}")
        plt.legend()
        save_plot(plt, f"{column}_{DAG}_distribution")  # Salva il grafico con il nome della variabile
        logging.info("---> Distribution plot saved for: %s", column)




def run_pipeline():
    """Main function to execute the exploratory data analysis pipeline."""
    logging.info("(----------------RUN PIPELINE------------------)")
    logging.info("Starting Exploratory Data Analysis (EDA)... ".upper())

    # Ottieni la lista di tutti i file CSV nella cartella Data_DAG

    

    # Itera su ogni file CSV nella cartella
    for filename in csv_files:
        filepath = os.path.join(DATA_PATH, filename)
        data = pd.read_csv(filepath)
        DAG=filename.split('_')[-1].replace('.csv', '')

        logging.info(f"Exploring {filename} dataset...")
        data_explorer(data=data, filename=filename, DAG=DAG)

        seed_data = data  

        logging.info(f"Plotting numeric distribution for {filename}...")
        plot_numeric_distributions(seed_data, filename, DAG)
            
if __name__ == "__main__":
    run_pipeline()
    logging.info("...Done!\n\n".upper())