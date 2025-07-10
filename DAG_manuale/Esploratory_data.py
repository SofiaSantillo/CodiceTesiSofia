import pandas as pd
import matplotlib.pyplot as plt
import logging
import io
import os
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
    logging.basicConfig(
        filename=os.path.join(root, path),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

LOG_PATH = "_Logs"
PLOT_DIR = "_Plot"
DATA_PATH = "Data_Droplet"
csv_file = "seed.csv"  # File singolo

setup_logging(LOG_PATH, "exploratory_data_analysis.log")

def save_plot(fig, plot_name):
    logging.info("(----------------SAVE PLOT------------------)")
    fig.savefig(f"{PLOT_DIR}/{plot_name}.png")
    plt.close()
    logging.info("---> Plot saved at: %s", f"{PLOT_DIR}/{plot_name}.png")


def data_explorer(data, filename, DAG):
    # Analisi ed esplorazione
    data.hist(bins=30, figsize=(12, 10))
    plt.tight_layout()
    plt.savefig(f"_Plot/histogram_{filename}.png")  # Assicurati che la cartella 'plots/' esista
    plt.close()

def plot_numeric_distributions(dataset, filename, DAG):
    logging.info("(----------------PLOT NUMERIC DISTRIBUTION------------------)")
    numeric_data = dataset.select_dtypes(include=['number']).dropna()
    for column in numeric_data.columns:
        plt.figure(figsize=(6, 4))
        sns.kdeplot(numeric_data[column], label=f"{column} Distribution", fill=True, color="skyblue")
        plt.title(f"Distribution of {column}")
        plt.legend()
        save_plot(plt, f"{column}_{DAG}_distribution")

def run_pipeline():
    logging.info("(----------------RUN PIPELINE------------------)")
    logging.info("Starting Exploratory Data Analysis (EDA)...")

    filepath = os.path.join(DATA_PATH, csv_file)
    data = pd.read_csv(filepath)
    DAG = csv_file.split('_')[-1].replace('.csv', '')

    logging.info(f"Exploring {csv_file} dataset...")
    data_explorer(data=data, filename=csv_file, DAG=DAG)

    logging.info(f"Plotting numeric distribution for {csv_file}...")
    plot_numeric_distributions(data, csv_file, DAG)

if __name__ == "__main__":
    run_pipeline()
    logging.info("...Done!\n\n")
