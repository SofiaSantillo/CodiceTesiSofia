import pandas as pd
import logging
from tabulate import tabulate
from probability_matrix_DAG23 import creation_matrix, creation_dataset
import numpy as np
import os
import json
from scipy.special import rel_entr



def cerca_prob(prob_array, frr_v, size_v, tfr_v):
    for item in prob_array:
        tripletta, prob = item
        frr, size, tfr = tripletta
        if frr == frr_v and size == size_v and tfr == tfr_v:
            return prob
    return 0.0

def probabilita_fattorizzata(matrix_frr_size, matrix_frr_tfr, prob_marginale_frr, prob_marginale_size, prob_empirica, logger, prob_marginale_tfr):
    array_probabilita_codizionata_size_dato_tfr = []
    array_probabilita_codizionata_tfr_dato_frr= []
    array_probabilita_empirica=[]
    log_data = []
    probabilita_fattorizzata=[]

    for j in range(1, matrix_frr_tfr.shape[1]):
        tfr_value=j-1
        print("tfr: ", tfr_value)
        for i in range(matrix_frr_tfr.shape[0]):
            frr_value=matrix_frr_tfr[i, 0]
            print("frr: ", frr_value)
            print("matrice frr tfr", matrix_frr_tfr[i,j], prob_marginale_frr[i])
            epsilon = 1e-8  
            den = matrix_frr_tfr[i, j] + epsilon
            p_codizionata_tfr_dato_frr= den/prob_marginale_frr[i] 
            array_probabilita_codizionata_tfr_dato_frr.append(p_codizionata_tfr_dato_frr)

            for k in range(1,  matrix_frr_size.shape[1]):
                size_value=k-1
                print("size: ", size_value)
                probabilità_congiunta=cerca_prob(prob_empirica, frr_value, size_value, tfr_value)
                array_probabilita_empirica.append(probabilità_congiunta)
                prob_condizionata_size= probabilità_congiunta/den
                array_probabilita_codizionata_size_dato_tfr.append(prob_condizionata_size)
                prob_fatt = prob_marginale_frr[i] * p_codizionata_tfr_dato_frr * prob_condizionata_size
                probabilita_fattorizzata.append(prob_fatt)
                
                log_data.append([frr_value, size_value, tfr_value, prob_marginale_frr[i],p_codizionata_tfr_dato_frr, prob_condizionata_size, prob_fatt, probabilità_congiunta])

    # Scrive i dati nel log in formato tabellare
    log_file = tabulate(log_data, headers=['FRR','SIZE','TFR','P(FRR)','P(TFR|FRR)', 'P(SIZE|FRR)', 'p. fattorizzata', 
                                            'p. empirica'], tablefmt='github')
    logger.info("\n" + log_file)

    return array_probabilita_empirica, probabilita_fattorizzata 

import numpy as np

def similarity_score(empirica, fattorizzata, logger=None):
    empirica = np.array(empirica)
    fattorizzata = np.array(fattorizzata)

    # Evita divisioni per zero sommando un piccolo epsilon
    epsilon = 1e-10
    somma = empirica + fattorizzata + epsilon
    diff = np.abs(empirica - fattorizzata)
    similarity = 1 - (diff / somma)

    mean_similarity = np.mean(similarity)

    if logger:
        logger.info(f"Similarità media: {mean_similarity:.4f}")

    return mean_similarity




def main():

    log_path = '_Logs'
    dataset_path = 'Data_DAG/Nodi_DAG2.csv' 
    DAG = "DAG2.3"
    log_file = os.path.join(log_path, 'confronto_prob_DAG2.3.log')

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    logger = logging.getLogger("confronto_prob_DAG2.3") 
    logger.setLevel(logging.INFO)

    logger.propagate = False

        # Aggiungi un handler solo se non già presente
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

  
    df = creation_dataset(log_path, dataset_path, DAG)
    matrix_frr_size, matrix_frr_tfr, p_marginale_frr, p_marginale_size, p_marginale_tfr, prob_empirica = creation_matrix(df)
    
    matrix_frr_size = np.array(matrix_frr_size)
    p_marginale_size=p_marginale_size.astype(float) 

    matrix_frr_tfr = np.array(matrix_frr_tfr, dtype=float)

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    
    empirica, fattorizzata=probabilita_fattorizzata(matrix_frr_size, matrix_frr_tfr, p_marginale_frr, p_marginale_size, prob_empirica, logger, p_marginale_tfr)

    avg_similarity = similarity_score(empirica, fattorizzata, logger)

    json_file = os.path.join(log_path, 'dag_avg_ratio_data.json')

    data = {
        'DAG': DAG,
        'avg_similarity': avg_similarity
    }

    # Leggi il file se esiste, altrimenti crea una lista vuota
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            try:
                all_data = json.load(f)
                if not isinstance(all_data, list):
                    all_data = []
            except json.JSONDecodeError:
                all_data = []
    else:
        all_data = []

    # Rimuovi entry esistente con stesso DAG
    all_data = [entry for entry in all_data if entry['DAG'] != data['DAG']]

    # Aggiungi la nuova entry (sovrascrivendo se esisteva)
    all_data.append(data)

    # Scrivi su file
    with open(json_file, 'w') as f:
        json.dump(all_data, f, indent=4)

    return avg_similarity

main()
    
    
    
    
    