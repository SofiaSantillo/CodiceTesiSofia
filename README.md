L'intero progetto si struttura nelle seguenti cartelle:
- _Data: contiene tutti i dataset che verranno utilizzati per l'implementazione del modello predittivo, della modellazione causale e del modello di Scale-Up. In particolare
    - seed_completo.csv contiene circa 300 formulazioni e rappresenta il dataset iniziale, fungendo da punto di partenza per l’addestramento dei modelli;
    - extension.csv, costituito da circa 50 formulazioni aggiuntive, che può essere impiegato sia per ampliare il numero di campioni di training sia come primo benchmark di validazione preliminare;
    - validation.csv rappresenta il vero set di validazione, il quale non deve mai essere utilizzato durante la fase di addestramento, garantendo così una stima imparziale delle prestazioni del modello sui dati non osservati;
    - data_1.csv, ottenuto unendo seed\_completo.csv ed extension.csv. Questo file rappresenta il dataset principale utilizzato nel progetto, contenente l’intera gamma di formulazioni disponibili per l’addestramento e la sperimentazione dei modelli;
    - data_1_Binning è il dataset binnato secondo quanto effettuato nella cartella 0_BINNING, utilizzato poi in fase di modellazione causale per il calcolo delle probabilità frequentiste;
    - dataset_ScaleUp.csv è il dataset fornito delle formulazione in fase di Scale Up industriale. Viene utilizzato in fase di check tra modello predittivo di machine learning puro e modello di ScaleUp creato tramite analisi causale, come validation set
    - dataset_ScaleUp_mixed_train.csv è il dataset composto unendo randomicamente formulazioni provenienti da data_1 e dataset_ScaleUp per il ri-allenamento del modello predittivo puro per confrontarne le prestazioni rispetto al modello di ScaleUp; rappresenta il training set
    - dataset_ScaleUp_mixed_validation.csv è il dataset composto unendo randomicamente formulazioni provenienti da data_1 e dataset_ScaleUp per il ri-allenamento del modello predittivo puro per confrontarne le prestazioni rispetto al modello di ScaleUp; rappresenta il validation set
-_File: contiene file di configurazione e strumenti ausiliari
-_Logs: contiene i log riferiti alle esecuzioni del modello predittivo puro
-_Model: cartella in cui sono salvati i modelli predittivi puri 
-_Plot: ...

PIPELINE: per favorire la leggibilità dell'intero progetto, le cartelle sono state numerate in ordine di esecuzione. In ognuna di essere sono presenti/possono essere presenti file di _Log, _Plot, _json ecc.. relativi alle esecuzioni degli script presenti nella medesima cartella
    - 0_BINNING: processo di discretizzazione del dataset data_1 per successivi calcoli delle probabilità frequentiste in fase di analisi causale. Eseguire solo lo script Binning_dataset.py

    - 1_MODEL: implementazione, addestramento ed esecuzione del modello rpedittivo di machine learning puro sul dataset data_1. Eseguire solo lo script Model_predictive.py

    - 2_DAG: creazione dello spazio dei dag a 3 nodi e espansione dei 10 dag a 3 nodi migliori con HillClimbing. Eseguire prima pipeline_3nDAG e poi HillClimbing

    - 3_D-SEPARATION: analisi e modifica dei 10 DAG costruiti con HillClimbing tramite conoscenza a priori, letteratura e conferma analisi e modifiche tramite d-separation. Eseguire: structure_analysis -> d-separation -> expanded_dag_correction

    - 4_SAMPLING: selezione dei 3 DAG migliori tra i 10 dag espansi modificati ai passi precedenti, tramite sampling: confronto tra dataset di partenza e dataset causale: 10 iterazioni dell'algoritmo per stima statistica dei migliori 3 dag (criterio: tradoff tra MAE (locale) e KS (globale)). Eseguire solo simulation_sampling.py

    - 5_ACE: confronto dei 3 dag selezionati dal sampling per scelta DAG ottimo. Il calcolo dell'ACE mi permette di decifrare edge migliori a discapito di altri e quindi scartare DAG meno potenti dal punto di vista causale. Eseguire "ACE" modificando la variabile DAG.

    - 6_VALIDAZIONE_INTERVENTISTA: Ricerca delle correlazioni spurie per i modelli rf, xgb_size, xgb_pdi. Eseguire solo "Research_spurious_correlations.py" modificando le flag a seconda di quello che ci serve (modificare in "Sensitivity_Shuffle.py" e "Shap_analysis.py" o "Shap_analysis_xgb.py" a seconda del modello che stiamo studiando). I risultati sono rispettivamente in "MASTER_LOG_rf.log" ".._xg_size.log", ".._xgb_pdi.log"

    
