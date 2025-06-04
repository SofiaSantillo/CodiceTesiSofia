## DATASET:
I dati DLS sono i dati in tabella nell'articolo "Microfluidic manufacturing of liposomes: Development and
optimization by design of experiment and machine learning."
Ho riportati questi ultimi in un file .csv tramite extraction_data_DLS in "Extraction_dataset".
Ho applicato a questi dati l'IQR (codice in Extraction_dataset/IQR_data_DLS.py). In particolare, il codice applica l'IQR a tutte le colonne numeriche;
salva in un file .png i grafici delle distribuzioni di ogni signola colonna prima e dopo l'applicazione dell'IQR; salva il nuovo dataset ripulito 
(elimato delle righe che avevano outlier per qualche features) nel file Data_DLS/data_DLS_cleaned.csv

