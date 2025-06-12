import logging
import os
import importlib.util



logging.basicConfig(filename='_Logs/Validation_Results.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Dizionario per memorizzare i moduli di validazione
moduli_val = {}

cartella_val = "Validation_DAG"

lower, upper = 0.9, 1.1

# Caricamento dei moduli di validazione
for file in os.listdir(cartella_val):
    if file.startswith("Validation_DAG") and file.endswith(".py"):
        nome_modulo_val = file[:-3]
        spec = importlib.util.spec_from_file_location(nome_modulo_val, os.path.join(cartella_val, file))
        modulo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(modulo)
        moduli_val[nome_modulo_val] = modulo


def run_pipeline():
    logging.info("\n\nINIZIO..")
    logging.info("Description | Percentage")

    previous_prefix = None



def run_pipeline():
    logging.info("\n\nINIZIO..")
    logging.info("Description | Percentage | DAG")

    previous_prefix = None
    dag_dir = "DAG_manuale"

    for nome_val, mod_val in moduli_val.items():
        if hasattr(mod_val, "percentage_well_done"):
            try:
                # Esegui la funzione senza log
                logging.getLogger().setLevel(logging.CRITICAL)
                percentage, description = mod_val.percentage_well_done(lower, upper)
                logging.getLogger().setLevel(logging.INFO)

                # Separatore se cambia DAG principale
                prefix = nome_val.split('.')[0]
                if previous_prefix is not None and prefix != previous_prefix:
                    logging.info("---------")
                previous_prefix = prefix

                # Costruisci path al file Python
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

                
                logging.info(f"{description} | {percentage:.2f}% | DAG: {dag_repr}")

            except Exception as e:
                logging.error(f"Errore in {nome_val}: {e}")


    
   
    logging.getLogger().setLevel(logging.INFO) 

    

if __name__ == "__main__":
    run_pipeline()
