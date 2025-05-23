DATASET:
I dati DLS sono i dati in tabella nell'articolo "Microfluidic manufacturing of liposomes: Development and
optimization by design of experiment and machine learning."
Ho riportati questi ultimi in un file .csv tramite extraction_data_DLS in "Extraction_dataset".
Ho applicato a questi dati l'IQR (codice in Extraction_dataset/IQR_data_DLS.py). In particolare, il codice applica l'IQR a tutte le colonne numeriche;
salva in un file .png i grafici delle distribuzioni di ogni signola colonna prima e dopo l'applicazione dell'IQR; salva il nuovo dataset ripulito 
(elimato delle righe che avevano outlier per qualche features) nel file Data_DLS/data_DLS_cleaned.csv

COSTRUZIONE DAG: Possibili metodi da implentare analizzati:
- CasualNex + NOTEARS: CausalNex è uno strumento utile se vuoi partire dai dati e costruire una rete causale, cioè un grafo che ti dice chi influenza chi. 
È pensato per chi lavora con dati tabellari (come CSV) e vuole capire le relazioni di causa-effetto in modo automatico.
Il cuore del processo è l’algoritmo NOTEARS, che è intelligente perché non prova tutte le combinazioni possibili, ma impara direttamente la struttura del grafo risolvendo un problema di ottimizzazione matematica. 
Scoperta automatica DAG --> ottimo per prototipi
- PGMpy(Probabilistic Graphical Models for Python): permette di lavorare con reti bayesiane, cioè strutture dove si può modellare non solo la causalità, ma anche la probabilità con cui certe cose accadono. 
Ti permette di fare "inferenza": puoi, per esempio, calcolare la probabilità che un evento accada sapendo che un altro è successo.
- Causal-learn: pensata per applicare algoritmi classici per scoprire la struttura causale a partire dai dati. 
Usa metodi solidi come il PC algorithm, FCI, GES e altri. Questi algoritmi ti permettono di costruire DAG anche in presenza di variabili latenti o non osservate.

SCELTA MODELLO COSTRUZIONE DAG: CausalNex + NOTEARS --> Ottimi trade-off tra semplicità e potenza

VALIDAZIONE DAG:
- Cross validation causale
- Simulazione di dati casuali (interventi) 
- Testing statistico delle dipendenza casuali- Stima della probabilità condizionale
- Verifica della coerenza con la distribuzione dei dati
- Validazione incrociata con modelli predittivi