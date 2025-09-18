import pandas as pd
import networkx as nx

# ---------------- INPUT ----------------
dag_edges = [('FRR', 'AQUEOUS'), ('AQUEOUS', 'PEG'), ('CHOL', 'ESM'), ('ESM', 'HSPC'), 
             ('AQUEOUS', 'TFR'), ('PEG', 'CHOL'), ('PEG', 'PDI'), ('ESM', 'SIZE'), 
             ('ESM', 'PDI'), ('TFR', 'PDI'), ('FRR', 'PDI'), ('TFR', 'SIZE'), ('FRR', 'SIZE')]

dag = nx.DiGraph(dag_edges)

feature_cols = ['FRR', 'AQUEOUS', 'PEG', 'CHOL', 'ESM', 'HSPC', 'TFR', 'SIZE', 'PDI']
target_nodes = ['SIZE', 'PDI']

# ---------------- Funzione per classificare le feature ----------------
def classify_feature_dag(feature: str, target: str, dag: nx.DiGraph) -> str:
    if feature == target:
        return 'target'
    all_preds = nx.ancestors(dag, target)
    if feature not in all_preds:
        return 'independent'
    direct_preds = list(dag.predecessors(target))
    if feature in direct_preds:
        return 'direct'
    return 'indirect'

# ---------------- Creazione dataframe ----------------
rows = []
for target in target_nodes:
    for feat in feature_cols:
        cat = classify_feature_dag(feat, target, dag)
        rows.append({
            'Target': target,
            'Feature': feat,
            'DAG_Category': cat
        })

df_dag_class = pd.DataFrame(rows)

# ---------------- Formattazione log tabellare ----------------
log_file_path = "6_VALIDAZIONE_INTERVENTISTA/_log/feature_dag_structures.log"
with open(log_file_path, "w") as f:
    # Header
    f.write("Target       | Feature         | DAG_Category\n")
    f.write("-------------|-----------------|-------------\n")
    
    # Righe
    for _, row in df_dag_class.iterrows():
        f.write(f"{row['Target']:<13}| {row['Feature']:<16}| {row['DAG_Category']:<12}\n")

print(f"Risultati salvati in '{log_file_path}' in forma tabellare.")
