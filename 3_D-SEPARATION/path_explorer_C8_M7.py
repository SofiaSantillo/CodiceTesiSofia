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

    def classify_path(self, path):
        """Classifica i percorsi come Catena, Collider, Forchetta o Arco diretto."""
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
        classified = [(p, self.classify_path(p)) for p in paths]
        return classified

def write_results_to_log(results, output_file, start, end):
    """Scrive i percorsi classificati in un file di log."""
    with open(output_file, "w") as f:
        f.write(f"{start} - {end} \n")
        for path, kind in results:
            f.write(f"Percorso: {path} -> {kind}\n")


if __name__ == "__main__":
    edges = [('AQUEOUS', 'PEG'), ('CHIP', 'PDI'), ('CHOL', 'ESM'), ('CHOL', 'PDI'), ('ESM', 'PDI'), ('ESM', 'SIZE'), ('FRR', 'CHOL'), ('FRR', 'PEG'), ('ML', 'ESM'), ('ML', 'HSPC'), ('ML', 'PEG'), ('TFR', 'CHOL')]

    dag = DAG(edges)
    start, end = "PDI", "SIZE"

    output_file = "3_D-SEPARATION/_logs/paths_C8_M7.log"

    results = dag.find_and_classify_paths(start, end)
    write_results_to_log(results, output_file, start, end)

