import importlib.util
import logging
import os
import json
import pickle
import numexpr




def load_dag_avg_similarity_data():
    json_file = os.path.join('_Logs', f'dag_avg_ratio_data.json')
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data
    else:
        return None

def run_pipeline():

    dag_dir = "DAG_manuale"
    selected_dags = []
    # Recupera i dati dal file JSON
    data = load_dag_avg_similarity_data()

    logging.info("\n\n--- INIZIO ---\n")

    if data:
        for entry in data:
            description = entry.get('DAG', 'Unknown')
            avg_similarity = entry.get('avg_similarity', 0.0)

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
                    if avg_similarity >= avg_threshold:
                        if description in ["DAG1.1", "DAG2.3"]:
                            continue
                        else:
                            selected_dags.append({
                                "description": description,
                                "percentage": avg_similarity,
                                "repr": dag_repr
                            })
                else:
                    dag_repr = "Funzione constraction_dag() non trovata"
            else:
                dag_repr = "File non trovato"

        if selected_dags:
            with open(f'_Logs/DAG_4_NODES/selected_dags_{avg_threshold}.pkl', 'wb') as f:
                pickle.dump(selected_dags, f)

            logging.info(f"\nSalvati {len(selected_dags)} DAG in selected_dags.pkl")

          
            logging.info("\n\n--- DAG SELEZIONATI SOPRA LA SOGLIA ---")
            for dag in selected_dags:
                descr = dag['description']
                perc = dag['percentage']
                repr_str = str(dag['repr'])
                # Limita lunghezza se la stringa è troppo lunga
                repr_str_short = repr_str if len(repr_str) < 300 else repr_str[:300] + '...'
                logging.info(f"{descr:<10} | {perc:<5.2f} | {repr_str_short}")
        else:
            print("Nessun DAG con percentuale ≥ 0.6 trovato.")
    else:
        print("Nessun dato trovato nel file JSON.")
    
    

if __name__ == "__main__":
    logging.getLogger('numexpr').setLevel(logging.CRITICAL)

    avg_threshold= 0.7
    
    logging.basicConfig(
        filename=f'_Logs/DAG_4_NODES/Selectioned_DAG_avg_threshold_{avg_threshold}.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    run_pipeline()
