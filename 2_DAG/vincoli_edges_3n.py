import json

not_allowed_edges = [
    #FRR
    ('PDI', 'FRR'),
    ('SIZE', 'FRR'),
    #TFR
    ('PDI', 'TFR'),
    ('SIZE', 'TFR'),
    #AQUE
    ('SIZE', 'AQUEOUS'),
    ('PDI', 'AQUEOUS'),
    #PEG
    ('PDI', 'PEG'),
    ('SIZE', 'PEG'), 
    #CHOL
    ('PDI', 'CHOL'),
    ('SIZE', 'CHOL'),
    #HSPC
    ('PDI', 'HSPC'),
    ('SIZE', 'HSPC'),
    #ESM
    ('PDI', 'ESM'),
    ('SIZE', 'ESM'),
    #CHIP
    ('PDI', 'CHIP'),
    ('SIZE', 'CHIP'),
    #ML
    ('PDI', 'ML'),
    ('SIZE', 'ML'),
    

    # ... aggiungi tutti gli altri archi che vuoi
]

# JSON non supporta tuple, quindi convertiamo le tuple in liste
not_allowed_edges_as_lists = [list(edge) for edge in not_allowed_edges]

# Salvataggio in file JSON
with open('2_DAG/_json/vincoli_edges_3n.json', 'w') as f:
    json.dump(not_allowed_edges_as_lists, f, indent=2)
