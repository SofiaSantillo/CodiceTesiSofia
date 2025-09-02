import json

with open('2_DAG/_json/dependency_results.json', 'r') as f:
    dep_dict = json.load(f)

with open('2_DAG/_json/similarity_3nDAG.json', 'r') as f:
    mini_dags = json.load(f)


def edge_valido(src, dst, dep_dict):
    #Controlla se l'edge src -> dst è valido secondo il file delle dipendenze
    if src in dep_dict:
        return dst in dep_dict[src]['variabili_influenti']
    return False


dag_validi_totali = {}
totale_validi = 0

for gruppo, dags in mini_dags.items():
    dag_validi = []
    for dag in dags:
        edges = dag['dag_edges']
        score = dag.get("score", 0)  # fallback 0 se non esiste

        # Check edges validi e score > 0.8
        if all(edge_valido(src, dst, dep_dict) for src, dst in edges) and score > 0.80:
            dag_validi.append(dag)

    if dag_validi:  
        dag_validi_totali[gruppo] = dag_validi
        print(f"Gruppo {gruppo}: {len(dag_validi)} DAG validi")
        totale_validi += len(dag_validi)
    

print(f"Totale DAG validi (tutti i gruppi): {totale_validi}")

with open('2_DAG/_json/3nDAG_filtered.json', 'w') as f:
    json.dump(dag_validi_totali, f, indent=4)
