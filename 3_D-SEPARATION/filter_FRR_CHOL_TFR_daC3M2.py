import re
from collections import defaultdict

class DAG:
    def __init__(self, edges):
        self.edges = edges
        self.graph = defaultdict(list)
        self.parents = defaultdict(list)
        for u, v in edges:
            self.graph[u].append(v)
            self.parents[v].append(u)

    def all_paths_valid(self, start, end):
        """Trova tutti i percorsi tra start e end senza ripetere nodi."""
        def dfs(node, path, visited):
            if node == end:
                all_paths.append(path)
                return
            for neighbor in self.graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, path + [neighbor], visited | {neighbor})
            for parent in self.parents[node]:
                if parent not in visited:
                    dfs(parent, path + [parent], visited | {parent})

        all_paths = []
        dfs(start, [start], {start})
        return all_paths

    def is_collider(self, path, node):
        """Verifica se il nodo è un collider lungo il percorso."""
        idx = path.index(node)
        if idx == 0 or idx == len(path) - 1:
            return False
        prev_node = path[idx - 1]
        next_node = path[idx + 1]
        # Collider se entrambi arrivano al nodo
        return (prev_node, node) in self.edges and (next_node, node) in self.edges

def parse_edges(edges_str):
    """Converti stringa di edges in lista di tuple."""
    return eval(edges_str)

def select_dags_with_chol_collider(log_file, start, end):
    with open(log_file) as f:
        content = f.read()

    dag_blocks = re.split(r"DAG #\d+", content)[1:]
    dag_numbers = [int(num) for num in re.findall(r"DAG #(\d+)", content)]

    selected_dags = []

    for dag_num, block in zip(dag_numbers, dag_blocks):
        edges_match = re.search(r"Edges:\s*(\[.*?\])", block, re.DOTALL)
        if not edges_match:
            continue
        edges = parse_edges(edges_match.group(1))
        dag = DAG(edges)

        paths = dag.all_paths_valid(start, end)
        # Condizione: unico percorso
        if len(paths) == 1:
            path = paths[0]
            # Contiene collider in CHOL?
            if 'CHOL' in path and dag.is_collider(path, 'CHOL'):
                selected_dags.append((dag_num, path))

    return selected_dags

if __name__ == "__main__":
    log_file = "3_D-SEPARATION/_logs/dag_structures.log"
    start, end = "FRR", "TFR"

    selected = select_dags_with_chol_collider(log_file, start, end)
    for dag_num, path in selected:
        print(f"DAG #{dag_num}: 1 percorso tra {start} e {end}")
        print(f"  {path}")

    dag_list = [dag_num for dag_num, _ in selected]
    print("\nLista dei DAG selezionati:", dag_list)