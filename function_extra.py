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

# Function to build the linear model
def build_linear_model(dag, dataset):
    linear_models = {}
    G = nx.DiGraph()
    
    for arc in dag.arcs:
        G.add_edge(*arc)

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
    """
    Converts a causaldag.DAG object into a NetworkX directed graph.

    Parameters:
        dag (causaldag.DAG): The DAG object containing arcs.

    Returns:
        networkx.DiGraph: Directed graph with the same structure as the DAG.
    """

    for arc in dag.arcs:
        G.add_edge(*arc)
    """
    Aggiunge tutti gli archi da un oggetto DAG a un oggetto networkx.DiGraph.

    Parametri:
        dag: oggetto con attributo .arcs (es. causaldag.DAG)
        G (nx.DiGraph): grafo esistente a cui aggiungere gli archi (opzionale).
                        Se None, viene creato un nuovo DiGraph.

    Ritorna:
        nx.DiGraph: grafo con gli archi del DAG.
    """

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