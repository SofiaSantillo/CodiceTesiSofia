import pandas as pd
import numpy as np
import logging

# Funzione di discretizzazione
def discretize(df, bins=4):
    return df.apply(lambda col: pd.cut(col, bins=bins, labels=False))

# Funzione per calcolare la probabilità fattorizzata secondo il DAG
def get_fact_prob(row, f_indep, f_dep1, f_dep2):
    tfr, size, pdi = row['TFR'], row['SIZE'], row['PDI']

    p_indep = f_indep.get(tfr, 0)
    p_dep1 = f_dep1.loc[tfr][size] if tfr in f_dep1.index else 0
    p_dep2 = f_dep2.loc[size][pdi] if size in f_dep2.index else 0

    # Probabilità fattorizzata secondo il DAG
    return p_indep * p_dep1 * p_dep2

# Funzione per calcolare la percentuale di probabilità ben spiegata dal modello
def percentage_well_done(lower=0.9, upper=1.1):
    """
    Calcola la percentuale della distribuzione congiunta ben spiegata dal modello
    secondo il DAG: FRR → SIZE → PDI.

    Parametri:
    - lower: soglia inferiore del ratio
    - upper: soglia superiore del ratio

    Ritorna:
    - percentuale di supporto ben spiegato
    - nome del DAG
    """

    # Impostazioni per il logging
    logging.basicConfig(filename='_Logs/validation_DAG1.log', level=logging.INFO, format='%(asctime)s - %(message)s')

    # Caricamento dati
    df = pd.read_csv("Data_Droplet/seed.csv", usecols=["TFR", "SIZE", "PDI"])

    # Discretizzazione delle variabili
    df_disc = discretize(df, bins=4)

    # Calcolo della distribuzione congiunta empirica
    joint = df_disc.value_counts(normalize=True).reset_index()
    joint.columns = ['TFR', 'SIZE', 'PDI', 'p_joint']

    # Calcolo delle distribuzioni marginali e condizionate secondo il DAG
    f_indep = df_disc['TFR'].value_counts(normalize=True)
    f_dep1 = df_disc.groupby('TFR')['SIZE'].value_counts(normalize=True).unstack().fillna(0)
    f_dep2 = df_disc.groupby('SIZE')['PDI'].value_counts(normalize=True).unstack().fillna(0)

    # Calcolo delle probabilità fattorizzate
    joint['p_fact'] = joint.apply(get_fact_prob, axis=1, f_indep=f_indep, f_dep1=f_dep1, f_dep2=f_dep2)
    
    # Calcolo del "ratio" tra distribuzione congiunta e probabilità fattorizzata
    joint['ratio'] = joint['p_joint'] / joint['p_fact'].replace(0, np.nan)

    # Log dei risultati principali
    logging.info(f"Top valori per ratio:\n{joint.sort_values('ratio', ascending=False).head(10)}")

    # Percentuale di distribuzione ben spiegata
    well_explained = joint[(joint['ratio'] >= lower) & (joint['ratio'] <= upper)]
    support_well_explained = well_explained['p_joint'].sum()
    support_total = joint['p_joint'].sum()

    percentage_well_explained = support_well_explained / support_total * 100

    logging.info(f"Percentuale di distribuzione ben spiegata (ratio ∈ [{lower}, {upper}]): {percentage_well_explained:.2f}%")

    DAG = "DAG1.2"
    return percentage_well_explained, DAG

