import ast
import networkx as nx
import pickle

def convert_log_to_graphs(log_path, out_path):
    graphs = {}
    current_dag = None
    edge_buffer = ""

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Nuovo DAG
            if line.startswith("#DAG"):
                if current_dag and edge_buffer:
                    edges = ast.literal_eval("[" + edge_buffer.rstrip(",") + "]")
                    G = nx.DiGraph()
                    G.add_edges_from(edges)
                    graphs[current_dag] = G

                # Nuovo nome DAG
                current_dag = line[1:]  # rimuove "#"
                edge_buffer = "" 

            elif line.startswith("GRUPPO"):
                if current_dag and edge_buffer:
                    edges = ast.literal_eval("[" + edge_buffer.rstrip(",") + "]")
                    G = nx.DiGraph()
                    G.add_edges_from(edges)
                    graphs[current_dag] = G
                current_dag = None
                edge_buffer = ""

            else:
                edge_buffer += line

        if current_dag and edge_buffer:
            edges = ast.literal_eval("[" + edge_buffer.rstrip(",") + "]")
            G = nx.DiGraph()
            G.add_edges_from(edges)
            graphs[current_dag] = G

    # Salva in pickle
    with open(out_path, "wb") as f:
        pickle.dump(graphs, f)


convert_log_to_graphs("Test_Leonardo_TesiSofia/_logs/Dag_base.log", "Test_Leonardo_TesiSofia/_logs/dag_nx.pkl")

with open("Test_Leonardo_TesiSofia/_logs/dag_nx.pkl", "rb") as f:
    graphs = pickle.load(f)

print("DAG trovati:", graphs.keys())
print("Archi DAG3:", graphs["DAG3"].edges())

import pickle

with open("Test_Leonardo_TesiSofia/_logs/dag_nx.pkl", "rb") as f:
    graphs = pickle.load(f)

print("DAG salvati:", list(graphs.keys()))
