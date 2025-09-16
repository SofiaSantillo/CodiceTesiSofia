import re
import networkx as nx

def parse_log_line(line):
    match = re.match(r"^([\d\.]+): (.+?) \|", line)
    if not match:
        return None, [], line  # ritorna la riga intera se non matcha
    dag_id = match.group(1)
    edges_str = match.group(2)
    edges = [tuple(edge.strip().split(" -> ")) for edge in edges_str.split(",")]
    return dag_id, edges, line


def build_graph(edges):
    G = nx.DiGraph()
    G.add_edges_from(edges)
    return G


def violates_constraints(G):

    # PEG - HSPC e PEG - ESM non devono avere cammini diretti tra loro
    for a, b in [("PEG", "HSPC"), ("HSPC", "PEG"), ("PEG", "ESM"), ("ESM", "PEG")]:
        if a in G.nodes and b in G.nodes:
            if G.has_edge(a, b):
                return True

    return False


def filter_and_write_log(input_file, output_file):
    """Filtra i DAG e scrive solo quelli validi in un nuovo file di log."""
    total_dags = 0
    valid_dags = 0
    with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            dag_id, edges, original_line = parse_log_line(line)
            if not dag_id:
                continue  
            total_dags += 1
            G = build_graph(edges)
            if not violates_constraints(G):
                f_out.write(original_line)
                valid_dags += 1

    print(f"DAG totali: {total_dags}")
    print(f"DAG validi salvati: {valid_dags}")
    print(f"File filtrato salvato in: {output_file}")


input_file = "2_DAG/_Logs/3nDAG_with_scores.log"    
output_file = "2_DAG/_Logs/3nDAG_valid_with_scores.log"  
filter_and_write_log(input_file, output_file)
