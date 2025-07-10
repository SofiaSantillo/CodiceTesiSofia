import importlib.util
import logging
import os
import json
import numexpr

logging.getLogger('numexpr').setLevel(logging.CRITICAL)

logging.basicConfig(filename='_Logs/Validation_Results.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_dag_avg_similarity_data():
    json_file = os.path.join('_Logs', 'dag_avg_ratio_data.json')
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data
    else:
        return None

def run_pipeline():
    logging.info("\n\nINIZIO..")
    logging.info("Description | Percentage | DAG")
    previous_prefix = None
    previous_dag_number = None 

    dag_dir = "DAG_manuale"

    # Recupera i dati dal file JSON
    data = load_dag_avg_similarity_data()

    if data:
        for entry in data:
            description = entry.get('DAG', 'Unknown')
            avg_similarity = entry.get('avg_similarity', 0.0)
            dag_number = description.split('.')[0]
          
            # Separatore se cambia il numero principale del DAG
            if previous_dag_number is not None and dag_number != previous_dag_number:
                logging.info("---------")
            
            previous_dag_number=dag_number

            dag_filename = f"Constraction_{description}.py"
            dag_filepath = os.path.join(dag_dir, dag_filename)
            dag_repr = "DAG non trovato"
            if os.path.exists(dag_filepath):
                # Import dinamico
                spec = importlib.util.spec_from_file_location(f"Constraction_{description}", dag_filepath)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Chiamata alla funzione che costruisce il DAG
                if hasattr(module, "constraction_dag"):
                    dag_repr = module.constraction_dag()
                else:
                    dag_repr = "Funzione constraction_dag() non trovata"
            else:
                dag_repr = "File non trovato"

            # Scrittura dei dati nel formato tabellare
            if description in ["DAG1.1", "DAG2.3"]:
                logging.info(f"[CANCELED] {description:<10} | {avg_similarity:<10.2f} | {dag_repr}")
            else:
                logging.info(f"{description:<10} | {avg_similarity:<10.2f} | {dag_repr}")
    else:
        logging.warning("Nessun dato trovato nel file JSON.")

if __name__ == "__main__":
    run_pipeline()
