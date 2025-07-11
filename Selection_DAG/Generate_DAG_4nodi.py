import pickle
import itertools
import networkx as nx
from tabulate import tabulate
import matplotlib.pyplot as plt

def plot_all_dags(pkl_file, plot_file):
    # Carica i DAG selezionati
    with open(pkl_file, 'rb') as f:
        selected_dags = pickle.load(f)

    # Crea una figura per contenere tutti i DAG
    fig, axes = plt.subplots(4, 4, figsize=(40, 40))
    axes = axes.flatten()

    # Plot di tutti i DAG in una griglia
    for i, dag in enumerate(selected_dags):
        print(dag)
        G = nx.DiGraph()  # Crea un grafo diretto

        edges = dag[1]
        for edge in edges:
            G.add_edge(*edge)

        ax = axes[i]
        pos = nx.spring_layout(G, seed=42) 
     
        nx.draw(G, pos, with_labels=True, ax=ax, node_color='lightblue', node_size=3000, font_size=10, font_weight='bold', edge_color='gray')

        ax.set_title(f'DAG {i+1}', fontsize=12)

    # Aggiungi un titolo generale
    plt.suptitle('Tutti i DAG', fontsize=16, fontweight='bold')

    # Regola la disposizione e salva l'immagine
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Aggiungi spazio per il titolo generale
    plt.savefig(plot_file, format='png')

import logging


def find_and_remove_duplicate_dags(pkl_file):
    with open(pkl_file, 'rb') as f:
        dags = pickle.load(f)

    seen = {}
    unique_dags = []
    duplicates = []

    for idx, dag in enumerate(dags):
        edges = dag[1]
        edge_set = frozenset(tuple(edge) for edge in edges)

        if edge_set in seen:
            duplicates.append((seen[edge_set], idx))
        else:
            seen[edge_set] = len(unique_dags)
            unique_dags.append(dag)

    if duplicates:
        print("DAG duplicati trovati e rimossi:")
        for dup in duplicates:
            print(f"DAG {dup[0]} è uguale a DAG {dup[1]}")
        
        # Salva solo DAG unici nel file originale
        with open(pkl_file, 'wb') as f:
            pickle.dump(unique_dags, f)
        print(f"\nTotale DAG dopo la rimozione: {len(unique_dags)}")
    else:
        print("Nessun DAG duplicato trovato.")


def filter_dags_and_save(table, pkl_file):

    filtered_dags = [dag for dag in table if "NO CORREL" not in dag[0] and "CICLO" not in dag[0]]

    # Salva il risultato in un file pickle
    with open(pkl_file, 'wb') as f:
        pickle.dump(filtered_dags, f)

    print(f"Filtrati {len(filtered_dags)} DAG e salvati in '{pkl_file}'.")
    return filtered_dags, pkl_file

def filter2_dags_and_save(table, pkl_file):
    filtered2_dags = [dag for dag in table if len(dag[1])==3]
    print(filtered2_dags)
    with open(pkl_file, 'wb') as f:
            pickle.dump(filtered2_dags, f)

    print(f"Filtrati {len(filtered2_dags)} DAG e salvati in '{pkl_file}'.")
    return filtered2_dags, pkl_file

def create_4nodesDags(threshold):

    with open(f'_Logs/DAG_4_NODES/selected_dags_{threshold}.pkl', 'rb') as f:
        selected_dags = pickle.load(f)

    dag_dict = {dag['description']: (dag['percentage'], dag['repr']) for dag in selected_dags}

    # Applica la trasformazione a tutti gli elementi del dizionario
    for key in dag_dict:
        score, edge_str = dag_dict[key]

        edges_list = edge_str.split(', ')
        
        # Converte ogni 'A -> B' in '(A,B)' e li concatena
        edges_tuples_str = ','.join(
            f"({src.strip()},{dst.strip()})"
            for src_dst in edges_list
            for src, dst in [src_dst.split('->')]
        )

        dag_dict[key] = (score, edges_tuples_str)


    def parse_edges(edge_str):
        edge_str = edge_str.strip()
        edge_str = edge_str.replace('),(', ')|(')  
        edge_str = edge_str.replace('(', '').replace(')', '')
        edge_pairs = edge_str.split('|')
        return [tuple(edge.strip().split(',')) for edge in edge_pairs]

    dag_with_4_nodes = []
    table = []

    for (key1, (score1, edges1)), (key2, (score2, edges2)) in itertools.combinations(dag_dict.items(), 2):
        edge_list1 = parse_edges(edges1)
        edge_list2 = parse_edges(edges2)


        combined_edges = list(set(edge_list1 + edge_list2))
        nodes = set([n for edge in combined_edges for n in edge])
        i=1
        if len(nodes) == 4:
            dag_info = ({
                'index': i,
                'combined_from': (key1, key2),
                'edges': combined_edges,
                'nodes': list(nodes),
                'avg_score': (score1 + score2) / 2
            })
            i=i+1
            # Filtraggio degli archi per evitare archi inversi
            filtered_edges = []
            seen_edges = set() 


            frr_tfr_causality = False
            size_pdi_causality = False
            has_duplicate = False  

            for edge in dag_info['edges']:
                if (edge[1], edge[0]) in seen_edges:
                    has_duplicate = True

                else:
                    filtered_edges.append(edge)
                    seen_edges.add(edge)
                if (edge == ('FRR', 'TFR') or edge == ('TFR', 'FRR')):
                    frr_tfr_causality = True

                if (edge == ('SIZE', 'PDI') or edge == ('PDI', 'SIZE')):
                    size_pdi_causality = True

            if has_duplicate:
                dag_info['combined_from'] = f"{dag_info['combined_from']} CICLO"

            if not frr_tfr_causality:
                dag_info['combined_from'] = f"{dag_info['combined_from']} + NO CORREL TRA FRR e TFR"

            if not size_pdi_causality:
                dag_info['combined_from'] = f"{dag_info['combined_from']} + NO CORREL TRA SIZE e PDI"

            table.append([dag_info['combined_from'], dag_info['edges'], dag_info['nodes'], dag_info['avg_score']])

        
        headers = ["Combinazione", "Archi", "Nodi", "Score Medio"]
        tabella_log= tabulate(table, headers=headers, tablefmt="github")
    logging.info("\n" + tabella_log)

    pkl_file=f'_Logs/DAG_4_NODES/all_dags_{threshold}.pkl'
    filter_dag, pkl_file=filter_dags_and_save(table, pkl_file)
    find_and_remove_duplicate_dags(pkl_file)
    
    plot_file = f'_Plot/DAG_4_NODES/all_dags_{threshold}.png'  
    plot_all_dags(pkl_file, plot_file)  

    filter2_dag, pkl_file=filter2_dags_and_save(filter_dag, pkl_file) 

    plot_file2 = f'_Plot/DAG_4_NODES/realistic_dags_{threshold}.png'  
    plot_all_dags(pkl_file, plot_file2) 

if __name__ == "__main__":
    threshold= 0.6

    logging.basicConfig(
        filename=f'_Logs/DAG_4_NODES/dag_4_nodes_{threshold}_threshold.log',
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        filemode='w' 
    )
    create_4nodesDags(threshold)