import pandas as pd
import os
import logging
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np
from itertools import product
import math
from Binning_FRR import bin_FRR
from Binning_TFR import bin_TFR
from Binning_PDI import bin_PDI


log_path = '_Logs'
dataset_path = 'Data_DAG/Nodi_DAG3.csv' 
plot_path="_Plot"
DAG = "DAG3.1"

if not os.path.exists(log_path):
    os.makedirs(log_path)

logging.basicConfig(filename=os.path.join(log_path, f'probability_{DAG}.log'), level=logging.INFO)
logger = logging.getLogger()

def setup_logger(name, log_file, level=logging.INFO):
    """Crea e restituisce un logger con handler separato"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.FileHandler(log_file, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

#Creazione del nuovo dataset con binning sulle variabili
def creation_dataset(log_path, dataset_path, DAG):

    logger_FRR = setup_logger('FRR', os.path.join(log_path, f'Binning_FRR.log'))
    logger_PDI = setup_logger('PDI', os.path.join(log_path, f'Binning_PDI.log'))
    logger_TFR = setup_logger('TFR', os.path.join(log_path, f'Binning_TFR.log'))

    logger.info(f"\n\n INIZIO\n\n\n")

    df = pd.read_csv(dataset_path)
    logger.info(f"Dataset originale:\n{df}\n")


    df_cleaned = df.dropna().copy()
    logger.info(f"Dataset dopo eliminazione righe con NaN:\n{df_cleaned}\n")

    df_cleaned = bin_FRR(df_cleaned, logger_FRR)
    df_cleaned = bin_TFR(df_cleaned, logger_TFR)
    df_cleaned = bin_PDI(df_cleaned, logger_PDI)


    # Crea il nuovo dataset
    df_new = df_cleaned.drop(columns=["TFR", "PDI", "FRR", "PDI_TS"])
    output_file = os.path.join('Data_DAG', f'new_dataset_{DAG}.csv')
    df_new.to_csv(output_file, index=False)
    logger.info(f"Dataset con variabili categorizzate:\n{df_new}\n")

    # Plotta istogrammi
    plot_vars = ['FRR_CAT', 'PDI_CAT', 'TFR_CAT']
    for var in plot_vars:
        plt.figure()
        df_cleaned[var].value_counts(sort=False).plot(kind='bar')
        plt.title(f"Istogramma {var}")
        plt.xlabel(var)
        plt.ylabel("Conteggio")
        plt.tight_layout()
        plt.savefig(os.path.join('_Plot', f"{var}_hist_binning.png"))
        plt.close()

    return df_new


def aggiungi_somme(matrix_copy):

    # Somme riga
    somme_righe = []
    for row in matrix_copy:
        somma_riga = sum(float(x) for x in row[1:])
        somme_righe.append(f"{somma_riga:.4f}")
        row.append(f"{somma_riga:.4f}")
    # Somme colonna
    somme_colonne = []
    for i in range(1, len(matrix_copy[0])):
        somma_col = sum(float(row[i]) for row in matrix_copy)
        somme_colonne.append(f"{somma_col:.4f}")
    return matrix_copy, somme_colonne, somme_righe #restituisce la matrice da graficare e le probabilità marginali

#calcolo probabilità marginali e congiunte di 2 variabili
def calcolo_matrice_probabilita(N, df, x, y):
    count = {}
    for _, row in df.iterrows():
        key = (row[x], row[y])       
        count[key] = count.get(key, 0) + 1

    
    x_values = sorted(df[x].unique())
    y_values = sorted(df[y].unique())
    
    matrix = []
    for tfr in x_values:
        row = [tfr]
        for pdi in y_values:
            prob = round(count.get((tfr, pdi), 0) / N, 4) #round(...,4) arrotonda a 4 cifre decimali
            row.append(prob)
        matrix.append(row)
    
    matrix_graph = [row[:] for row in matrix]
    matrix_graph, somme_colonne, somme_righe = aggiungi_somme(matrix_graph)
    matrix_graph.append([f"P({y}=j)"] + somme_colonne)
    
    headers = [f"{x} / {y}"] + [f"{s}" for s in y_values] + [f"P({x}=i)"]
    title=(f"Matrice P({x}, {y})")
    logger.info(f"\n{title}")
    logger.info(tabulate(matrix_graph, headers=headers, tablefmt="pretty", floatfmt=".4f"))
    return matrix, somme_colonne[:-1], somme_righe



def calcolo_matrice_probabilita_congiunta_totale(N, df, x, y, z):
    count_frr_pdi_tfr = {}

    # Ottieni tutti i valori unici delle tre variabili
    frr_values = df[f"{x}"].unique()
    pdi_values = df[f"{y}"].unique()
    tfr_values = df[f"{z}"].unique()

    # Inizializza tutte le combinazioni possibili con count = 0
    for combo in product(frr_values, pdi_values, tfr_values):
        count_frr_pdi_tfr[combo] = 0

    # Conta le occorrenze effettive
    for i in range(N):
        row = df.iloc[i]
        frr = row[f"{x}"]
        pdi = row[f"{y}"]
        tfr = row[f"{z}"]
        key = (frr, pdi, tfr)
        count_frr_pdi_tfr[key] += 1

    # Calcola le probabilità
    prob_array = []
    for key, count in count_frr_pdi_tfr.items():
        prob = count / N
        prob_array.append([key, prob])  # [ (FRR, PDI, TFR), Probabilità ]

    # Log dei risultati
    logger.info(f"\nP({x}, {y}, {z}):")
    logger.info(tabulate(prob_array, headers=[f"({x}, {y}, {z})", "Probabilità"], tablefmt="pretty"))

    return prob_array



def creation_matrix(df):
    N = len(df)

    matrix_frr_pdi, prob_marginale_pdi1, prob_marginale_frr1=calcolo_matrice_probabilita(N, df, x='FRR_CAT', y='PDI_CAT')
    matrix_tfr_pdi, prob_marginale_pdi2, prob_marginale_tfr1=calcolo_matrice_probabilita(N, df, x='TFR_CAT', y='PDI_CAT')
    matrix_frr_tfr, prob_marginale_tfr2, prob_marginale_frr2=calcolo_matrice_probabilita(N, df, x='FRR_CAT', y='TFR_CAT')

    probabilità_congiunta_tot=calcolo_matrice_probabilita_congiunta_totale(N, df, x='FRR_CAT', y='PDI_CAT', z='TFR_CAT')

    prob_marginale_pdi1 = np.array(prob_marginale_pdi1, dtype=float)
    prob_marginale_pdi2 = np.array(prob_marginale_pdi2, dtype=float)
    prob_marginale_frr1 = np.array(prob_marginale_frr1, dtype=float)
    prob_marginale_frr2 = np.array(prob_marginale_frr2, dtype=float)
    prob_marginale_tfr1 = np.array(prob_marginale_tfr1, dtype=float)
    prob_marginale_tfr2 = np.array(prob_marginale_tfr2, dtype=float)
  

    if not np.allclose(prob_marginale_pdi1, prob_marginale_pdi2, atol=1e-3):
        logger.error("Le probabilita' marginali PDI calcolate da diverse matrici non corrispondono!")
    else:
        logger.info("Le probabilita' marginali PDI corrispondono.")
        prob_marginale_pdi=prob_marginale_pdi1
        
    if not np.allclose(prob_marginale_frr1, prob_marginale_frr2, atol=1e-3):
        logger.error("Le probabilita' marginali FRR calcolate da diverse matrici non corrispondono!")
    else:
        logger.info("Le probabilita' marginali FRR corrispondono.")
        prob_marginale_frr=prob_marginale_frr1

    if not np.allclose(prob_marginale_tfr1, prob_marginale_tfr2, atol=1e-3):
        logger.error("Le probabilita' marginali TFR calcolate da diverse matrici non corrispondono!")
    else:
        logger.info("Le probabilita' marginali TFR corrispondono.")
        prob_marginale_tfr=prob_marginale_tfr1
    
    return matrix_frr_pdi, matrix_frr_tfr, matrix_tfr_pdi, prob_marginale_frr, prob_marginale_pdi, prob_marginale_tfr, probabilità_congiunta_tot




if __name__ == "__main__":
    df=creation_dataset(log_path, dataset_path, DAG="DAG3.1")
    creation_matrix(df)