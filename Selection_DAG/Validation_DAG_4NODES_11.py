import pandas as pd
import logging
from tabulate import tabulate
from Probability_matrix_DAG_4NODES import creation_matrix, creation_dataset
import numpy as np
import os
import json
from scipy.special import rel_entr



def cerca_prob(prob_array, frr_v, size_v, pdi_v, tfr_v):
    for item in prob_array:
        quadr, prob = item
        frr, size, pdi, tfr = quadr
        if frr == frr_v and size == size_v and pdi == pdi_v and tfr == tfr_v:
            return prob
    return 0.0

def probabilita_fattorizzata(matrix_frr_tfr, matrix_pdi_frr, matrix_size_pdi, prob_marginale_tfr, prob_marginale_frr, prob_marginale_pdi, prob_empirica, logger):
    array_probabilita_codizionata_frr_dato_tfr = []
    array_probabilita_codizionata_pdi_dato_frr= []
    array_probabilita_codizionata_size_dato_pdi= []
    array_probabilita_empirica=[]
    log_data = []
    probabilita_fattorizzata=[]


    for i in range(matrix_frr_tfr.shape[0]):
        frr_value=i
        print("frr:", frr_value)
        for k in range(1, matrix_frr_tfr.shape[1]):
            tfr_value=k-1
            print("tfr: ", tfr_value)
            print(matrix_frr_tfr[i,k], prob_marginale_tfr[k-1] )
            p_codizionata_frr_dato_tfr= matrix_frr_tfr[i,k]/ prob_marginale_tfr[k-1]
            array_probabilita_codizionata_frr_dato_tfr.append(p_codizionata_frr_dato_tfr)

            for j in range(matrix_pdi_frr.shape[0]): 
                pdi_value=j
                print("pdi: ", pdi_value)
                print(matrix_pdi_frr[j,i+1],prob_marginale_frr[i])
                p_condizionata_pdi_dato_frr=matrix_pdi_frr[j,i+1]/prob_marginale_frr[i]
                array_probabilita_codizionata_pdi_dato_frr.append(p_condizionata_pdi_dato_frr)

                for n in range(matrix_size_pdi.shape[0]):
                    size_value=n
                    print("size: ", size_value)
                    print(matrix_size_pdi[n,j+1], prob_marginale_pdi[j])
                    prob_condizionata_size= matrix_size_pdi[n,j+1]/prob_marginale_pdi[j]
                    array_probabilita_codizionata_size_dato_pdi.append(prob_condizionata_size)

                    prob_fatt = prob_marginale_tfr[k-1] * p_codizionata_frr_dato_tfr * p_condizionata_pdi_dato_frr * prob_condizionata_size
                    probabilita_fattorizzata.append(prob_fatt)




                    probabilità_congiunta=cerca_prob(prob_empirica, frr_value, size_value, pdi_value, tfr_value)
                    array_probabilita_empirica.append(probabilità_congiunta)
                    log_data.append([frr_value, size_value, pdi_value, tfr_value, prob_marginale_tfr[k-1], p_codizionata_frr_dato_tfr, p_condizionata_pdi_dato_frr, prob_condizionata_size, prob_fatt, probabilità_congiunta])

    # Scrive i dati nel log in formato tabellare
    log_file = tabulate(log_data, headers=['FRR','SIZE','PDI','TFR', 'P(TFR)','P(FRR|TFR)','P(PDI|FRR)', 'P(SIZE|PDI)', 'p. fattorizzata', 
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

    log_path = '_Logs/DAG_4_NODES'
    dataset_path = 'Data_DAG/Nodi_DAG_4NODES.csv' 
    DAG = "DAG_4NODES_11"
    log_file = os.path.join(log_path, 'confronto_prob_DAG_4NODES_11.log')

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    logger = logging.getLogger("confronto_prob_DAG_4NODES_11") 
    logger.setLevel(logging.INFO)

    logger.propagate = False

    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

  
    df = creation_dataset(log_path, dataset_path, DAG)
    matrix_frr_tfr, matrix_frr_size, matrix_size_pdi, matrix_tfr_pdi, matrix_size_tfr, matrix_pdi_frr, p_marginale_tfr, p_marginale_frr, p_marginale_size, p_marginale_pdi, prob_empirica = creation_matrix(df)
    
    matrix_frr_tfr = np.array(matrix_frr_tfr)
    matrix_frr_size = np.array(matrix_frr_size)
    matrix_size_pdi = np.array(matrix_size_pdi)
    matrix_pdi_frr = np.array(matrix_pdi_frr)

    p_marginale_tfr=np.array(p_marginale_tfr, dtype=float)
    p_marginale_frr=np.array(p_marginale_frr, dtype=float)
    p_marginale_size=np.array(p_marginale_size, dtype=float)
    p_marginale_pdi=np.array(p_marginale_pdi, dtype=float)

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    
    empirica, fattorizzata=probabilita_fattorizzata(matrix_frr_tfr, matrix_pdi_frr, matrix_size_pdi, p_marginale_tfr, p_marginale_frr, p_marginale_pdi, prob_empirica, logger)
      
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
    
    
    
    
    