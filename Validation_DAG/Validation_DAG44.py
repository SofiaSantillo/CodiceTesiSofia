import pandas as pd
import logging
from tabulate import tabulate
from probability_matrix_DAG44 import creation_matrix, creation_dataset
import numpy as np
import os
import json
from scipy.special import rel_entr



def cerca_prob(prob_array, tfr_v, size_v, pdi_v):
    for item in prob_array:
        tripletta, prob = item
        tfr, size, pdi = tripletta
        if tfr == tfr_v and size == size_v and pdi == pdi_v:
            return prob
    return 0.0

def probabilita_fattorizzata(matrix_tfr_size, matrix_size_pdi, prob_marginale_tfr, prob_marginale_pdi, prob_empirica, logger):
    array_probabilita_codizionata_size_dato_tfr = []
    array_probabilita_codizionata_size_dato_pdi= []
    array_probabilita_empirica=[]
    log_data = []
    probabilita_fattorizzata=[]


    for i in range(1, matrix_tfr_size.shape[1]):
        tfr_value=i-1
        print("tfr: ", tfr_value)
        for j in range(matrix_tfr_size.shape[0]):
            size_value=j
            print("size: ", size_value)
            print(matrix_tfr_size[j,i], prob_marginale_tfr[i-1])
            p_codizionata_size_dato_tfr= matrix_tfr_size[j,i]/prob_marginale_tfr[i-1] 
            array_probabilita_codizionata_size_dato_tfr.append(p_codizionata_size_dato_tfr)

            for k in range(1, matrix_size_pdi.shape[1]): 
                pdi_value=k-1
                print("pdi: ", pdi_value)
                print(matrix_size_pdi[j,k], prob_marginale_pdi[k-1])
                prob_condizionata_pdi= matrix_size_pdi[j,k]/prob_marginale_pdi[k-1]
                array_probabilita_codizionata_size_dato_pdi.append(prob_condizionata_pdi)
                prob_fatt = prob_marginale_tfr[i-1] * prob_marginale_pdi[k-1]* p_codizionata_size_dato_tfr * prob_condizionata_pdi
                probabilita_fattorizzata.append(prob_fatt)
                probabilità_congiunta=cerca_prob(prob_empirica, tfr_value, size_value, pdi_value)
                array_probabilita_empirica.append(probabilità_congiunta)
                log_data.append([tfr_value, size_value, pdi_value, prob_marginale_tfr[i-1], prob_marginale_pdi[k-1], p_codizionata_size_dato_tfr, prob_condizionata_pdi, prob_fatt, probabilità_congiunta])

    # Scrive i dati nel log in formato tabellare
    log_file = tabulate(log_data, headers=['TFR','SIZE','PDI','P(TFR)','P(PDI)','P(SIZE|TFR)', 'P(SIZE|PDI)', 'p. fattorizzata', 
                                            'p. empirica'], tablefmt='github')
    logger.info("\n" + log_file)

    return array_probabilita_empirica, probabilita_fattorizzata 

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
    DAG = "DAG4.4"
    log_file = os.path.join(log_path, 'confronto_prob_DAG4.4.log')

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    logger = logging.getLogger("confronto_prob_DAG4.4") 
    logger.setLevel(logging.INFO)

    logger.propagate = False

    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

  
    df = creation_dataset(log_path, dataset_path, DAG)
    matrix_tfr_size, matrix_pdi_size, p_marginale_tfr, p_marginale_pdi, prob_empirica = creation_matrix(df)
    
    matrix_tfr_size = np.array(matrix_tfr_size)
    matrix_pdi_size = np.array(matrix_pdi_size)
    p_marginale_pdi=p_marginale_pdi.astype(float) 
    p_marginale_tfr=p_marginale_tfr.astype(float) 
    matrix_pdi_size = np.array(matrix_pdi_size, dtype=float)

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    
    empirica, fattorizzata=probabilita_fattorizzata(matrix_tfr_size, matrix_pdi_size, p_marginale_tfr, p_marginale_pdi, prob_empirica, logger)
      
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

    all_data = [entry for entry in all_data if entry['DAG'] != data['DAG']]

    # Aggiungi la nuova entry (sovrascrivendo se esisteva)
    all_data.append(data)

    with open(json_file, 'w') as f:
        json.dump(all_data, f, indent=4)
    
    print(avg_similarity)

    return avg_similarity


main()
    
    
    
    
    