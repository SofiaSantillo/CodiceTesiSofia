import ast
import sys
import networkx as nx
import itertools
import pandas as pd

sys.stdout = open("5_ACE/_logs/ACE_DAG3.log", "w")

with open('5_ACE/_logs/DAG3.log', 'r') as f:
    content = f.read()
edges = ast.literal_eval(content)
G = nx.DiGraph()
G.add_edges_from(edges)

data = pd.read_csv('Data_Droplet/seed_Binning_ordinato.csv')

def compute_ACE_general_reference(P_do):
    """
    Calcola l'ACE generale usando come riferimento il primo valore di X.
    """
    x_vals = sorted(P_do.keys()) 
    ref = x_vals[0]
    E_ref = sum(y * P_do[ref][y] for y in P_do[ref])
    total = 0
    for x_val in x_vals[1:]:
        E_x = sum(y * P_do[x_val][y] for y in P_do[x_val])
        total += E_x - E_ref
    return total / (len(x_vals) - 1) if len(x_vals) > 1 else 0

for Y in G.nodes():
    direct_parents = list(G.predecessors(Y)) 
    if not direct_parents:
        continue
    print(f"\nGenitori diretti di {Y}: {direct_parents}")

    for X in direct_parents:
        print(f"\n--- Calcolo P({Y} | do({X})) e ACE generale ---")
        
        PA = list(G.predecessors(X))
        P_do = {}
        
        if not PA:
            # X senza genitori → formula semplificata
            for x_val in data[X].unique():
                P_do[x_val] = {}
                for y_val in data[Y].unique():
                    prob = len(data[(data[Y]==y_val) & (data[X]==x_val)]) / len(data[data[X]==x_val])
                    P_do[x_val][y_val] = prob
                    print(f"Formula: P({Y}={y_val} | do({X}={x_val})) = P({Y}={y_val} | {X}={x_val})")
                    print(f"Valore: {prob}")
                    
        else:
            # X ha genitori → do-calculus con somma pesata
            PA_values = [data[pa].unique() for pa in PA]
            combinations = list(itertools.product(*PA_values))

            for x_val in data[X].unique():
                P_do[x_val] = {}
                for y_val in data[Y].unique():
                    terms = []  
                    total = 0
                    for z in combinations:
                        
                        pa_filter = pd.Series([True] * len(data))
                        pa_str = []
                        for idx, pa in enumerate(PA):
                            pa_filter &= (data[pa] == z[idx])
                            pa_str.append(f"{pa}={z[idx]}")
                        
                        # Numero di osservazioni con PA = z
                        pa_count = len(data[pa_filter])
                        if pa_count == 0:
                            continue
                        
                        # Probabilità P(PA = z)
                        P_pa = pa_count / len(data)
                        
                        # Stima P(Y=y | X=x, PA=z)
                        denominator = len(data[(data[X] == x_val) & pa_filter])
                        if denominator == 0:
                            continue
                        numerator = len(data[(data[Y] == y_val) & (data[X] == x_val) & pa_filter])
                        P_y_given_x_pa = numerator / denominator
                        
                        contrib = P_y_given_x_pa * P_pa
                        total += contrib
                        
                        terms.append(f"P({Y}={y_val} | {X}={x_val}, {', '.join(pa_str)}) * P({', '.join(pa_str)})")
                    
                    formula_str = " + ".join(terms)
                    print(f"Formula: P({Y}={y_val} | do({X}={x_val})) = {formula_str}")
                    print(f"Valore: {total}")
                    
                    P_do[x_val][y_val] = total
        
        ace_ref = compute_ACE_general_reference(P_do)
        print(f"\nACE generale({X} -> {Y}) con riferimento = {ace_ref}")
