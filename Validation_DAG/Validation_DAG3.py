import pandas as pd
import numpy as np
import logging

# Funzione di discretizzazione
def discretize(df, bins=4):
    return df.apply(lambda col: pd.cut(col, bins=bins, labels=False))

# Funzione per calcolare la probabilità fattorizzata secondo il DAG
def get_fact_prob(row, f_indep, f_dep1, f_dep2):
    tfr, frr, pdi = row['TFR'], row['FRR'], row['PDI']
    
    p_indep = f_indep.get(tfr, 0)
    p_dep1 = f_dep1.loc[tfr][frr] if tfr in f_dep1.index else 0
    p_dep2 = f_dep2.loc[frr][pdi] if frr in f_dep2.index else 0
    
    return p_indep * p_dep1 * p_dep2

# ✅ Funzione principale da chiamare esternamente
def percentage_well_done(lower, upper):
    # Caricamento dati e calcoli interni alla funzione
    df = pd.read_csv("Data_Droplet/seed.csv", usecols=["TFR", "FRR", "PDI"])
    df_disc = discretize(df, bins=4)

    # Calcolo distribuzioni
    joint = df_disc.value_counts(normalize=True).reset_index()
    joint.columns = ['TFR', 'FRR', 'PDI', 'p_joint']

    f_indep = df_disc['TFR'].value_counts(normalize=True)
    f_dep1 = df_disc.groupby('TFR')['FRR'].value_counts(normalize=True).unstack().fillna(0)
    f_dep2 = df_disc.groupby('FRR')['PDI'].value_counts(normalize=True).unstack().fillna(0)

    joint['p_fact'] = joint.apply(get_fact_prob, axis=1, f_indep=f_indep, f_dep1=f_dep1, f_dep2=f_dep2)
    joint['ratio'] = joint['p_joint'] / joint['p_fact'].replace(0, np.nan)

    well_explained = joint[(joint['ratio'] >= lower) & (joint['ratio'] <= upper)]
    support_well_explained = well_explained['p_joint'].sum()
    support_total = joint['p_joint'].sum()

    percentage_well_explained = support_well_explained / support_total * 100

    return percentage_well_explained, "DAG3"
