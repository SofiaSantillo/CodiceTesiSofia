import pandas as pd
import numpy as np
import logging

logging.basicConfig(filename='_Logs/validation_DAG1.1.log', level=logging.INFO, format='%(asctime)s - %(message)s')


def discretize(df, bins=4):
    return df.apply(lambda col: pd.cut(col, bins=bins, labels=False))


df = pd.read_csv("Data_DAG/Nodi_DAG1.csv")
df_disc = discretize(df, bins=4)

columns = ['FRR', 'SIZE', 'PDI']
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
joint.columns = ['FRR', 'SIZE', 'PDI', 'p_joint']

# Calcola le distribuzioni marginali e condizionate
f_frr = df_disc['FRR'].value_counts(normalize=True)
f_size_given_frr = df_disc.groupby('FRR')['SIZE'].value_counts(normalize=True).unstack().fillna(0)
f_pdi_given_frr_and_size = df_disc.groupby(['FRR', 'SIZE'])['PDI'].value_counts(normalize=True).unstack().fillna(0)

# Calcola la probabilità fattorizzata secondo il DAG
def get_fact_prob(row):
    frr, size, pdi = row['FRR'], row['SIZE'], row['PDI']
    

    p_frr = f_frr.get(frr, 0)
    p_size_given_frr = f_size_given_frr.loc[frr][size] if frr in f_size_given_frr.index else 0
    p_pdi_given_frr_and_size = f_pdi_given_frr_and_size.loc[(frr, size)][pdi] if (frr, size) in f_pdi_given_frr_and_size.index else 0
    
    # Probabilità fattorizzata secondo il DAG
    return p_frr * p_size_given_frr * p_pdi_given_frr_and_size

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
