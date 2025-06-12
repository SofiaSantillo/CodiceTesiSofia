import pandas as pd
import numpy as np
import logging

# Impostazioni per il logging
logging.basicConfig(filename='_Logs/validation_DAG2.3.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Funzione di discretizzazione
def discretize(df, bins=4):
    return df.apply(lambda col: pd.cut(col, bins=bins, labels=False))

# Funzione per calcolare la probabilità fattorizzata secondo il DAG
def get_fact_prob(row, f_indep, f_dep1, f_dep2):
    frr, tfr, size = row['FRR'], row['TFR'], row['SIZE']

    p_indep = f_indep.get(frr, 0)
    p_dep1 = f_dep1.loc[frr][tfr] if frr in f_dep1.index else 0
    p_dep2 = f_dep2.loc[(frr, tfr)][size] if (frr, tfr) in f_dep2.index else 0

    return p_indep * p_dep1 * p_dep2

# Espone solo questa funzione
def percentage_well_done(lower=0.9, upper=1.1):
    # Caricamento dati
    df = pd.read_csv("Data_Droplet/seed.csv", usecols=["FRR", "TFR", "SIZE"])

    # Discretizzazione
    df_disc = discretize(df, bins=4)

    # Distribuzioni marginali e condizionate secondo DAG: FRR → TFR → SIZE
    f_indep = df_disc['FRR'].value_counts(normalize=True)
    f_dep1 = df_disc.groupby('FRR')['TFR'].value_counts(normalize=True).unstack().fillna(0)
    f_dep2 = df_disc.groupby(['FRR', 'TFR'])['SIZE'].value_counts(normalize=True).unstack().fillna(0)

    # Distribuzione congiunta empirica
    joint = df_disc.value_counts(normalize=True).reset_index()
    joint.columns = ['FRR', 'TFR', 'SIZE', 'p_joint']

    # Calcolo probabilità fattorizzata
    joint['p_fact'] = joint.apply(get_fact_prob, axis=1, f_indep=f_indep, f_dep1=f_dep1, f_dep2=f_dep2)
    joint['ratio'] = joint['p_joint'] / joint['p_fact'].replace(0, np.nan)

    # Percentuale di ratio ∈ [lower, upper]
    well_explained = joint[(joint['ratio'] >= lower) & (joint['ratio'] <= upper)]
    support_well_explained = well_explained['p_joint'].sum()
    support_total = joint['p_joint'].sum()
    percentage = support_well_explained / support_total * 100

    return percentage, "DAG2.3"
