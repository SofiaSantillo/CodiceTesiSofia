import pandas as pd
import logging
from tabulate import tabulate
from probability_matrix_DAG4 import creation_matrix, creation_dataset
import numpy as np
import os
import json
from scipy.special import rel_entr



def cerca_prob(prob_array, size_v, pdi_v, tfr_v):
    for item in prob_array:
        tripletta, prob = item
        size, pdi, tfr = tripletta
        if size == size_v and pdi == pdi_v and tfr == tfr_v:
            return prob
    return 0.0

def probabilita_fattorizzata(matrix_size_pdi, matrix_tfr_pdi, prob_marginale_size, prob_marginale_tfr, prob_empirica, logger):
    array_probabilita_codizionata_pdi_dato_tfr = []
    array_probabilita_codizionata_pdi_dato_size= []
    array_probabilita_empirica=[]
    log_data = []
    probabilita_fattorizzata=[]

    for i in range(matrix_tfr_pdi.shape[0]):
        tfr_value=i
        print(matrix_tfr_pdi)
        print(tfr_value)
        for j in range(1, matrix_tfr_pdi.shape[1]):
            pdi_value=j-1
            print(pdi_value)
            print("elem pdi dato tfr", matrix_tfr_pdi[i,j], prob_marginale_tfr[i])
            p_codizionata_pdi_dato_tfr= matrix_tfr_pdi[i,j]/prob_marginale_tfr[i] 
            array_probabilita_codizionata_pdi_dato_tfr.append(p_codizionata_pdi_dato_tfr)

            for k in range(matrix_size_pdi.shape[0]):
                size_value=k
                print("size:", size_value)
                print("elem pdi dato size", matrix_size_pdi[k,j], prob_marginale_size[k])
                prob_condizionata_pdi= matrix_size_pdi[k,j]/prob_marginale_size[k]
                array_probabilita_codizionata_pdi_dato_size.append(prob_condizionata_pdi)
                prob_fatt = prob_marginale_tfr[i] * prob_marginale_size[k] * p_codizionata_pdi_dato_tfr * prob_condizionata_pdi
                probabilita_fattorizzata.append(prob_fatt)
                probabilità_congiunta=cerca_prob(prob_empirica, size_value, pdi_value, tfr_value)
                array_probabilita_empirica.append(probabilità_congiunta)
                log_data.append([size_value, pdi_value, tfr_value, prob_marginale_tfr[i],p_codizionata_pdi_dato_tfr, prob_condizionata_pdi, prob_fatt, probabilità_congiunta])

    # Scrive i dati nel log in formato tabellare
    log_file = tabulate(log_data, headers=['SIZE','PDI','TFR','P(TFR)','P(PDI|TFR)', 'P(PDI|SIZE)', 'p. fattorizzata', 
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
    dataset_path = 'Data_DAG/Nodi_DAG4.csv' 
    DAG = "DAG4"
    log_file = os.path.join(log_path, 'confronto_prob_DAG4.log')

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    logger = logging.getLogger("confronto_prob_DAG4") 
    logger.setLevel(logging.INFO)

    logger.propagate = False

        # Aggiungi un handler solo se non già presente
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

  
    df = creation_dataset(log_path, dataset_path, DAG)
    matrix_size_pdi, matrix_tfr_pdi, p_marginale_size, p_marginale_tfr,prob_empirica = creation_matrix(df)
    
    matrix_size_pdi = np.array(matrix_size_pdi)
    p_marginale_size=p_marginale_size.astype(float) 
    p_marginale_tfr=p_marginale_tfr.astype(float) 
    matrix_tfr_pdi = np.array(matrix_tfr_pdi, dtype=float)

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    
    empirica, fattorizzata=probabilita_fattorizzata(matrix_size_pdi, matrix_tfr_pdi, p_marginale_size, p_marginale_tfr, prob_empirica, logger)

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
    
    
    
    
    