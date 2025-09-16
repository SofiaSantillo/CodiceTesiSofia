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
   
]

not_allowed_edges_as_lists = [list(edge) for edge in not_allowed_edges]

with open('2_DAG/_json/vincoli_edges_3n.json', 'w') as f:
    json.dump(not_allowed_edges_as_lists, f, indent=2)
