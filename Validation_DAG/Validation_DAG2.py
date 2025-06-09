import pandas as pd
import numpy as np
import logging


logging.basicConfig(filename='_Logs/validation_DAG2.log', level=logging.INFO, format='%(asctime)s - %(message)s')


def discretize(df, bins=4):
    return df.apply(lambda col: pd.cut(col, bins=bins, labels=False))


df = pd.read_csv("Data_DAG/Nodi_DAG2.csv")
df_disc = discretize(df, bins=4)

columns = ['FRR', 'TFR', 'SIZE']
n_bins = 4
bin_edges = {}

# Discretizzazione e salvataggio intervalli
for col in columns:
    df[f'{col}_bin'], bins = pd.cut(df[col], bins=n_bins, labels=False, retbins=True, include_lowest=True)
    bin_edges[col] = bins

    logging.info(f"\n{col} bins:")
    for i in range(len(bins)-1):
        logging.info(f"  Bin {i}: [{bins[i]:.2f}, {bins[i+1]:.2f})")

# Calcola la distribuzione congiunta empirica
joint = df_disc.value_counts(normalize=True).reset_index()
joint.columns = ['FRR', 'TFR', 'SIZE', 'p_joint']

# Calcola le distribuzioni marginali e condizionate
f_tfr = df_disc['TFR'].value_counts(normalize=True)
f_frr_given_tfr = df_disc.groupby('TFR')['FRR'].value_counts(normalize=True).unstack().fillna(0)
f_size_given_frr = df_disc.groupby('FRR')['SIZE'].value_counts(normalize=True).unstack().fillna(0)

# Calcola la probabilità fattorizzata secondo il DAG
def get_fact_prob(row):
    frr, tfr, size = row['FRR'], row['TFR'], row['SIZE']
    
    p_tfr = f_tfr.get(frr, 0)
    p_frr_given_tfr = f_frr_given_tfr.loc[tfr][frr] if tfr in f_frr_given_tfr.index else 0
    p_size_given_frr = f_size_given_frr.loc[frr][size] if frr in f_size_given_frr.index else 0
    
    # Probabilità fattorizzata secondo il DAG
    return p_tfr * p_frr_given_tfr * p_size_given_frr

joint['p_fact'] = joint.apply(get_fact_prob, axis=1)

joint['ratio'] = joint['p_joint'] / joint['p_fact'].replace(0, np.nan)
logging.info(f"RESULTS:\n{joint.sort_values('ratio', ascending=False).head(10)}")

# Calcolo della percentuale di probabilità ben spiegata dal modello
lower, upper = 0.9, 1.1  
well_explained = joint[(joint['ratio'] >= lower) & (joint['ratio'] <= upper)]
support_well_explained = well_explained['p_joint'].sum()
support_total = joint['p_joint'].sum()
percentage_well_explained = support_well_explained / support_total * 100

logging.info(f"\nPercentuale di distribuzione ben spiegata (ratio ∈ [{lower}, {upper}]): {percentage_well_explained:.2f}%")
