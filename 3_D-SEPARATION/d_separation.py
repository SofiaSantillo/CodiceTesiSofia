import logging
import os
import re
import pandas as pd
from collections import defaultdict
from itertools import combinations
import numpy as np

# -------------------- CLASSE DAG --------------------
class DAG:
    def __init__(self, edges):
        self.edges = edges
        self.graph = defaultdict(list)
        self.parents = defaultdict(list)
        for u, v in edges:
            self.graph[u].append(v)
            self.parents[v].append(u)

    def all_paths_valid(self, start, end):
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

    def classify_path(self, path):
        if len(path) == 2:
            u, v = path
            if (u, v) in self.edges or (v, u) in self.edges:
                return "Arco diretto"
        elif len(path) == 3:
            a, b, c = path
            if (a, b) in self.edges and (b, c) in self.edges:
                return "Catena"
            elif (a, b) in self.edges and (c, b) in self.edges:
                return "Collider"
            elif (b, a) in self.edges and (b, c) in self.edges:
                return "Forchetta"
        return "Struttura piu' lunga"

    def find_and_classify_paths(self, start, end):
        paths = self.all_paths_valid(start, end)
        return [(p, self.classify_path(p)) for p in paths]


# -------------------- PARSING FILE --------------------
def parse_equivalence_classes(file_path):
    """Ritorna un dizionario {classe: [lista DAG]}"""
    classes = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            match = re.match(r"Classe #(\d+): \[(.*?)\]", line.strip())
            if match:
                cls = int(match.group(1))
                dags = [int(x) for x in match.group(2).split(",")]
                classes[cls] = dags
    return classes


def parse_dag_structures(file_path):
    """Ritorna un dizionario {dag_id: edges}"""
    dags = {}
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    blocks = content.split("DAG #")
    for block in blocks[1:]:
        lines = block.strip().splitlines()
        dag_id = int(lines[0].strip())
        edges_line = [l for l in lines if l.startswith("Edges:")][0]
        edges_str = edges_line.replace("Edges:", "").strip()
        edges = eval(edges_str)
        dags[dag_id] = edges
    return dags


# -------------------- ENTROPIA & CMI --------------------
def joint_entropy(arrays):
    if isinstance(arrays, np.ndarray):
        if arrays.ndim == 1:
            arrays = [arrays]
        elif arrays.ndim == 2:
            arrays = [arrays[:, i] for i in range(arrays.shape[1])]
    arrays = [np.asarray(a).ravel() for a in arrays]
    df = pd.DataFrame(arrays).T
    counts = df.value_counts(normalize=True)
    return -np.sum(counts * np.log2(counts))

def conditional_mutual_information(x, y, z=None):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if z is not None and len(z) > 0:
        z = [np.asarray(col).ravel() for col in z]
        XZ = np.column_stack([x] + z)
        YZ = np.column_stack([y] + z)
        XYZ = np.column_stack([x, y] + z)
        Z_only = np.column_stack(z)
        return joint_entropy(XZ) + joint_entropy(YZ) - joint_entropy(XYZ) - joint_entropy(Z_only)
    else:
        XY = np.column_stack([x, y])
        return joint_entropy(x) + joint_entropy(y) - joint_entropy(XY)

def compute_cmi_paths(paths, df, x_col, y_col, epsilon=1e-5):
    from itertools import chain, combinations

    # 1. Trova tutte le variabili intermedie globali da tutti i path
    all_intermediates = set()
    normalized_paths = []
    for path in paths:
        if path[0] != x_col or path[-1] != y_col:
            path = path[::-1]
        normalized_paths.append(path)
        all_intermediates.update(path[1:-1])
    all_intermediates = list(all_intermediates)

    # Funzione per generare il powerset
    def powerset(iterable):
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

    results = []
    for path in normalized_paths:
        # 2. Genera tutte le combinazioni possibili di variabili intermedie globali
        for cond_set in powerset(all_intermediates):
            if cond_set:
                combined = pd.concat([df[x_col], df[y_col]] + [df[col] for col in cond_set], axis=1).dropna()
                x_vals_clean = combined[x_col].values
                y_vals_clean = combined[y_col].values
                z_vals_clean = [combined[col].values for col in cond_set]
                cmi = conditional_mutual_information(x_vals_clean, y_vals_clean, z_vals_clean)
            else:
                combined = pd.concat([df[x_col], df[y_col]], axis=1).dropna()
                x_vals_clean = combined[x_col].values
                y_vals_clean = combined[y_col].values
                cmi = conditional_mutual_information(x_vals_clean, y_vals_clean, None)

            results.append({
                "path": path,
                "conditioned_on": cond_set if cond_set else None,
                "cmi": cmi,
                "d_separated": cmi < epsilon
            })

    return pd.DataFrame(results)


# -------------------- LOGGING --------------------
def setup_logging(log_file_path):
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file_path, mode="w", encoding="utf-8"),
                  logging.StreamHandler()]
    )

def write_paths_and_cmi_to_log(results, cmi_df, output_file, dag_id, start, end):
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(f"\n================ DAG #{dag_id} =================\n")
        f.write(f"Percorsi {start} -> {end}\n")
        for path, kind in results:
            f.write(f"  Percorso: {path} -> {kind}\n")
        f.write(f"\nRisultati CMI {start} -> {end}\n")
        for _, row in cmi_df.iterrows():
            f.write(
                f"  Path: {row['path']} | Cond: {row['conditioned_on']} | "
                f"CMI: {row['cmi']:.6f} | D-Separato: {row['d_separated']}\n"
            )

# -------------------- MAIN --------------------
if __name__ == "__main__":
    eq_file = "3_D-SEPARATION/_logs/equivalence_classes.log"
    dag_file = "3_D-SEPARATION/_logs/dag_structures.log"
    data_file = "_Data/data_1.csv"
    log_dir = os.path.join("3_D-SEPARATION", "_logs")

    os.makedirs(log_dir, exist_ok=True)

    df = pd.read_csv(data_file)
    variables = list(df.columns)

    eq_classes = parse_equivalence_classes(eq_file)
    dag_structs = parse_dag_structures(dag_file)

    for cls, dag_list in eq_classes.items():
        dag_id = dag_list[0]
        if dag_id not in dag_structs:
            print(f"DAG #{dag_id} non trovato in {dag_file}")
            continue
        edges = dag_structs[dag_id]

        log_file_path = os.path.join(log_dir, f"d-separation_dag{dag_id}.log")
        setup_logging(log_file_path)
        logger = logging.getLogger(f"d-separation-{dag_id}")

        dag = DAG(edges)

        for start, end in combinations(variables, 2):
            results = dag.find_and_classify_paths(start, end)

            if not results: 
                continue
            cmi_df = compute_cmi_paths([p for p, _ in results], df, x_col=start, y_col=end)
            write_paths_and_cmi_to_log(results, cmi_df, log_file_path, dag_id, start, end)