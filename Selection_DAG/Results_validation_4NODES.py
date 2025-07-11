import importlib.util
import logging
import os
import json
import pickle
import numexpr

logging.getLogger('numexpr').setLevel(logging.CRITICAL)

logging.basicConfig(filename='_Logs/DAG_4_NODES/Validation_Results_4NODES.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_dag_percentage_data():
    json_file = os.path.join('_Logs/DAG_4_NODES', 'dag_avg_ratio_data.json')
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data
    else:
        return None

def run_pipeline():
    logging.info("\n\nINIZIO..")
    logging.info("Description | Avg_similarity | DAG")
    previous_dag_number = None 

    dag_dir = "_Logs/DAG_4_NODES"
    data = load_dag_percentage_data()

    if data:
        dag_filename = "all_dags_0.6.pkl"
        dag_filepath = os.path.join(dag_dir, dag_filename)

        all_dags = []
        if os.path.exists(dag_filepath):
            with open(dag_filepath, 'rb') as f:
                all_dags = pickle.load(f)
        i=0
        for entry in data:
            description = entry.get('DAG', 'Unknown')
            avg_similarity = entry.get('avg_similarity', 0.0)
            dag_number = description.split('.')[0]

            if previous_dag_number is not None and dag_number != previous_dag_number:
                logging.info("---------")
            previous_dag_number = dag_number

            # Costruzione lista degli avg_score
            avg_similarity_composition = []
            for dag in all_dags:
                avg_similarity_composition.append(dag[3])

            # Logging
            logging.info(f"{description:<10} | {avg_similarity:<10.2f} | {avg_similarity_composition[i]}")
            i=i+1
            print(i)
           
    else:
        logging.warning("Nessun dato trovato nel file JSON.")

if __name__ == "__main__":
    run_pipeline()
