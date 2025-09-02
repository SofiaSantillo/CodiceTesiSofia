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

    def has_collider(self, collider):
        """Controlla se il DAG contiene un collider specifico (a->b<-c)."""
        a, b, c = collider
        return (a, b) in self.edges and (c, b) in self.edges

def parse_edges(edges_str):
    return eval(edges_str)

def write_dags_with_collider(log_file, collider_to_find, output_file):
    with open(log_file) as f:
        content = f.read()

    dag_blocks = re.split(r"(DAG #\d+)", content)[1:] 
    dags_to_write = []

    for i in range(0, len(dag_blocks), 2):
        dag_header = dag_blocks[i].strip()
        dag_num = int(re.search(r"\d+", dag_header).group())
        dag_content = dag_blocks[i+1]

        edges_match = re.search(r"Edges:\s*(\[.*?\])", dag_content, re.DOTALL)
        if not edges_match:
            continue

        edges = parse_edges(edges_match.group(1))
        dag = DAG(edges)

        if dag.has_collider(collider_to_find):
            dags_to_write.append(f"{dag_header}\n{dag_content}")

    with open(output_file, "w") as f:
        f.writelines(dags_to_write)

    print(f"{len(dags_to_write)} DAG scritti in {output_file}.")

if __name__ == "__main__":
    log_file = "3_D-SEPARATION/_logs/dag_structures.log"
    output_file = "3_D-SEPARATION/_logs/dag_structures2.log"
    collider_to_find = ("ESM", "ML", "HSPC")

    write_dags_with_collider(log_file, collider_to_find, output_file)

