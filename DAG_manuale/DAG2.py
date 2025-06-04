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


def analyze_dag(dag):

    G = nx.DiGraph()
    for arc in dag.arcs:
        G.add_edge(*arc)

    forks = set()
    colliders = set()
    chains = []
    backdoor_paths = []

    nodes = list(G.nodes)

    for A in nodes:
        for B in nodes:
            for C in nodes:
                if len({A, B, C}) < 3:
                    continue

                # Collider: A → B ← C (avoid duplicates)
                if G.has_edge(A, B) and G.has_edge(C, B):
                    if A < C:
                        colliders.add((A, B, C))
                    else:
                        colliders.add((C, B, A))

                # Fork: A ← B → C (avoid duplicates)
                if G.has_edge(B, A) and G.has_edge(B, C):
                    if A < C:
                        forks.add((A, B, C))
                    else:
                        forks.add((C, B, A))

                # Chain: A → B → C (all valid triplets)
                if G.has_edge(A, B) and G.has_edge(B, C):
                    chains.append((A, B, C))

    # Backdoor paths
    for A in nodes:
        for B in nodes:
            if A != B and not G.has_edge(A, B):
                if nx.has_path(G, A, B):
                    backdoor_paths.append((A, B))
    
    def format_triplets(triplets, arrows):
        return [f"{t[0]} {arrows[0]} {t[1]} {arrows[1]} {t[2]}" for t in triplets]

    def format_pairs(pairs):
        return [f"{a} ⇒ {b}" for a, b in pairs]

    structure_file = "DAG_manuale/dag2_structure.txt"
    # Write to file
    with open(structure_file, "w", encoding="utf-8") as f:
        f.write("DAG2 STRUCTURE:\n\n\n")
        # Writing collider information
        f.write("Colliders (A → B ← C):\n")
        f.write("\n".join(format_triplets(colliders, ('→', '←'))) or "None")
        f.write("\n\n")
    
        # Writing fork information
        f.write("Forks (A ← B → C):\n")
        f.write("\n".join(format_triplets(forks, ('←', '→'))) or "None")
        f.write("\n\n")
    
        # Writing chain information
        f.write("Chains (A → B → C):\n")
        f.write("\n".join(format_triplets(chains, ('→', '→'))) or "None")
        f.write("\n\n")
    
        # Writing backdoor paths information
        f.write("Backdoor paths (A ⇒ B):\n")
        f.write("\n".join(format_pairs(backdoor_paths)) or "None")
        f.write("\n")

    return list(forks), list(colliders), chains, backdoor_paths


# Function to build the linear model
def build_linear_model(dag, dataset):
    linear_models = {}
    G = nx.DiGraph()
    
    for arc in dag.arcs:
        G.add_edge(*arc)

    # One-hot encoding for 'CHIP'
    if 'CHIP' in dataset.columns:
        dataset = pd.get_dummies(dataset, columns=['CHIP'], drop_first=True)

    imputer = SimpleImputer(strategy='mean')

    for node in G.nodes:
        if node not in dataset.columns:
            continue
        
        predecessors = list(G.predecessors(node))
        valid_predecessors = [p for p in predecessors if p in dataset.columns]

        if valid_predecessors:
            X = dataset[valid_predecessors].copy()
            y = dataset[node].copy()

            # Remove rows with NaN values
            data = pd.concat([X, y], axis=1).dropna()
            X_clean = data[valid_predecessors]
            y_clean = data[node]

            X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.3, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            linear_models[node] = (model, r2)
    
    return linear_models


# Function to build the nonlinear model
def build_nonlinear_model(dag, dataset):
    nonlinear_models = {}
    G = nx.DiGraph()

    for arc in dag.arcs:
        G.add_edge(*arc)

    # One-hot encoding for 'CHIP'
    if 'CHIP' in dataset.columns:
        dataset = pd.get_dummies(dataset, columns=['CHIP'], drop_first=True)

    imputer = SimpleImputer(strategy='mean')

    for node in G.nodes:
        if node not in dataset.columns:
            continue

        predecessors = [p for p in G.predecessors(node) if p in dataset.columns]

        if predecessors:
            X = dataset[predecessors].copy()
            y = dataset[node].copy()

            # Imputation of NaNs
            X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
            y_imputed = y.fillna(y.mean())

            X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_imputed, test_size=0.3, random_state=42)

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            nonlinear_models[node] = (model, r2)
            print(f"Model for {node} built with R²: {r2}")

    return nonlinear_models


# Function to compare models and write to file
def compare_models(dag, dataset, log_filename):
    # Build linear model
    linear_models = build_linear_model(dag, dataset)
    
    # Build nonlinear model
    nonlinear_models = build_nonlinear_model(dag, dataset)
    
    # Create log file
    with open(log_filename, "w") as log_file:
        log_file.write("Comparison between linear and nonlinear models:\n\n")
        
        # Write linear and nonlinear models with R²
        log_file.write("Linear models (R²):\n")
        for node, (model, r2) in linear_models.items():
            log_file.write(f"Node: {node}, Linear Model: {model}, R²: {r2}\n")
        
        log_file.write("\nNonlinear models (R²):\n")
        for node, (model, r2) in nonlinear_models.items():
            log_file.write(f"Node: {node}, Nonlinear Model: {model}, R²: {r2}\n")
        
        log_file.write("\nFinal comparison between models:\n")
        
        # Comparison using R² score
        for node in dag.nodes:
            linear_r2 = linear_models[node][1] if node in linear_models else None
            nonlinear_r2 = nonlinear_models[node][1] if node in nonlinear_models else None
            
            if linear_r2 and nonlinear_r2:
                if linear_r2 > nonlinear_r2:
                    log_file.write(f"\nFor node {node}, the linear model is better with R²: {linear_r2} vs {nonlinear_r2} of the nonlinear model.")
                else:
                    log_file.write(f"\nFor node {node}, the nonlinear model is better with R²: {nonlinear_r2} vs {linear_r2} of the linear model.")
            elif linear_r2:
                log_file.write(f"\nFor node {node}, only the linear model is available with R²: {linear_r2}.")
            elif nonlinear_r2:
                log_file.write(f"\nFor node {node}, only the nonlinear model is available with R²: {nonlinear_r2}.")
    return


# === DAG Construction ===
df = pd.read_csv("Data_DAG/Nodi_DAG2.csv") 
column_names = df.columns.tolist()
column_names.append("mt")

dag = DAG(set(column_names))
dag.add_arc('CHIP', 'SIZE')
dag.add_arc('CHIP', 'PDI')
dag.add_arc('FRR', 'SIZE')
dag.add_arc('FRR', 'PDI')
dag.add_arc('TFR', 'SIZE')
dag.add_arc('TFR', 'PDI')
dag.add_arc('TFR', 'FRR')
dag.add_arc('SIZE', 'PDI')


# Create networkx graph from dag.arcs
G = nx.DiGraph()
for arc in dag.arcs:
    G.add_edge(*arc)

# Plot and save the DAG
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=2000, font_size=10, font_weight='bold', arrows=True)
plt.savefig('_Plot/dag2_output.png', format='png')
plt.close()

# Analyze structures
forks, colliders, chains, backdoor_paths = analyze_dag(dag)
log_filename = '_Logs/eq_strutturate_dag2.log'

# Compare models
compare_models(dag, df, log_filename)
