## Data_DLS: non ancora utilizzati
I dati DLS sono i dati in tabella nell'articolo "Microfluidic manufacturing of liposomes: Development and
optimization by design of experiment and machine learning."
Ho riportati questi ultimi in un file .csv tramite extraction_data_DLS in "Extraction_dataset".
Ho applicato a questi dati l'IQR (codice in Extraction_dataset/IQR_data_DLS.py). In particolare, il codice applica l'IQR a tutte le colonne numeriche;
salva in un file .png i grafici delle distribuzioni di ogni signola colonna prima e dopo l'applicazione dell'IQR; salva il nuovo dataset ripulito 
(elimato delle righe che avevano outlier per qualche features) nel file Data_DLS/data_DLS_cleaned.csv

## Data_DAG
Mi creo file con solo i nodi che mi servono per costruire i vari "mmini-DAG" tramite la funzione Extraction_node.py"

## DAG_manuale
-   Constaction_DAG.py: creo il "mini-DAG" manualmente, salvo nella cartella "_Structure_manual_DAG" le strutture base, ossia fork, collider, backdoor, chain, dei singoli dag (un file per ognuno)
-   Exploratory_data.py: ho definito le funzioni data_explorer e plot_numeric_distribution prendendole dalla cartella "tesi_sofia" di GitHub --> penso che potrebbero essermi utili in futuro (salvo le distribuzioni, gli istogrammi e i risultati dell'esecuzione del codice rispettivamente in file png in _Plot e nel file di log in _Logs)

## Validation_DAG
costruisco un file di validazione per ogni "mini-DAG", in cui:
    - discretizzo le variabili in 4 bin
    - calcolo da distribuzione congiunta empirica "p_joint" delle variabili discretizzate: 
        p_joint(A=a,B=b,C=c)= Numero totale di osservazioni/Numero di occorrenze di (A=a, B=b, C=c)
    - calcolo le distribuzioni marginali e condizionate
    - calcola la probabilità fattorizzata "p_fact" secondo il modello DAG
    - confronto la distribuzione congiunta empirica con quella fattorizzata calcolando il rapporto "ratio"
    - calcola la percentuale di probabilità che il modello fattorizzato spiega bene (con ratio tra 0,9 e 1,1)


