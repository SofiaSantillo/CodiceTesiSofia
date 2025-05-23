import pandas as pd
from causalnex.structure.notears import from_pandas
import matplotlib.pyplot as plt
import networkx as nx

# Load the data from the CSV file
df = pd.read_csv('Data_DLS/data_DLS.csv')
df['SIZE'] = pd.to_numeric(df['SIZE'], errors='coerce')
df_numeric = df.dropna(subset=['SIZE'])
df_numeric = df.select_dtypes(include=['number'])
df_numeric = df_numeric.dropna()

# Build the DAG with NOTEARS
dag = from_pandas(df_numeric)
G = nx.DiGraph()
G.add_nodes_from(dag.nodes)
for edge in dag.edges:
    G.add_edge(edge[0], edge[1])
node_colors = []

for node in G.nodes:
    if node == 'SIZE':
        node_colors.append('lightgreen')  
    elif node == 'PDI':
        node_colors.append('lightcoral') 
    else:
        node_colors.append('skyblue')  

# Draw the DAG using networkx with different colors for SIZE and PDI
plt.figure(figsize=(10, 8)) 
nx.draw(G, with_labels=True, node_size=3000, node_color=node_colors, font_size=10, font_weight="bold", edge_color="gray")
plt.title("DAG Generated with CausalNex")

# Save the image as PNG
plt.savefig("_Images/dag_Causalnex_data_DLS.png", format="PNG")

plt.close()  

with open("_Log/causal_dependencies_data_DLS.log", "w") as log_file:
    log_file.write("Causal dependencies of features\n\n")  # Add header to log file
    for edge in dag.edges:
        log_file.write(f"{edge[0]} --> {edge[1]}\n")
