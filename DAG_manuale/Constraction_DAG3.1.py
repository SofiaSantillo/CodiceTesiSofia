import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from causaldag import DAG
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import os
from sklearn.impute import SimpleImputer



# === DAG Construction ===
def constraction_dag():
    df = pd.read_csv("Data_Droplet/seed.csv") 
    column_names = df.columns.tolist()

    dag = DAG(set(column_names))
    dag.add_arc('FRR', 'TFR')
    dag.add_arc('TFR', 'PDI')


    # Create networkx graph from dag.arcs
    G = nx.DiGraph()
    for arc in dag.arcs:
        G.add_edge(*arc)

    # Plot and save the DAG
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=5000, font_size=15, font_weight='bold', arrows=True, arrowsize=30)
    plt.savefig('_Plot/DAG3.1.png', format='png')
    plt.close()

  
    edges_text = [f"{src} -> {dst}" for src, dst in G.edges()]
    dag_repr = ", ".join(edges_text)  

    return dag_repr

if __name__ == "__main__":
    constraction_dag()





