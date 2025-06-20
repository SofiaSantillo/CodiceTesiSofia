import pandas as pd
import numpy as np
import logging

# Funzione di discretizzazione
def discretize(df, bins=4):
    return df.apply(lambda col: pd.cut(col, bins=bins, labels=False))

# Funzione per calcolare la probabilità fattorizzata secondo il DAG
def get_fact_prob(row, f_indep, f_dep1, f_dep2):
    tfr, frr, size = row['TFR'], row['FRR'], row['SIZE']
    
    p_indep = f_indep.get(tfr, 0)
    p_dep1 = f_dep1.loc[tfr][frr] if tfr in f_dep1.index else 0
    p_dep2 = f_dep2.loc[frr][size] if frr in f_dep2.index else 0

    return p_indep * p_dep1 * p_dep2

# Funzione per calcolare la percentuale di probabilità ben spiegata dal modello (DAG2)
def percentage_well_done(lower=0.9, upper=1.1):
    logging.basicConfig(filename='_Logs/validation_DAG2.log', level=logging.INFO, format='%(asctime)s - %(message)s')

    df = pd.read_csv("Data_Droplet/seed.csv", usecols=["TFR", "FRR", "SIZE"])
    df_disc = discretize(df, bins=4)

    # Calcolo delle distribuzioni
    f_indep = df_disc['TFR'].value_counts(normalize=True)
    f_dep1 = df_disc.groupby('TFR')['FRR'].value_counts(normalize=True).unstack().fillna(0)
    f_dep2 = df_disc.groupby('FRR')['SIZE'].value_counts(normalize=True).unstack().fillna(0)

    # Calcolo distribuzione congiunta
    joint = df_disc.value_counts(normalize=True).reset_index()
    joint.columns = ['TFR', 'FRR', 'SIZE', 'p_joint']

    # Probabilità fattorizzata e ratio
    joint['p_fact'] = joint.apply(get_fact_prob, axis=1, f_indep=f_indep, f_dep1=f_dep1, f_dep2=f_dep2)
    joint['ratio'] = joint['p_joint'] / joint['p_fact'].replace(0, np.nan)

    # Percentuale ben spiegata
    well_explained = joint[(joint['ratio'] >= lower) & (joint['ratio'] <= upper)]
    support_well_explained = well_explained['p_joint'].sum()
    support_total = joint['p_joint'].sum()
    percentage_well_explained = support_well_explained / support_total * 100

    DAG = "DAG2"
    return percentage_well_explained, DAG

