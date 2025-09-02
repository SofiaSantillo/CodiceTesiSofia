import numpy as np
import pandas as pd
import math

def binning_terrel_scott(data, column):
    n = data[column].dropna().shape[0]
    k = math.ceil((2 * n) ** (1/3))
    return k

def bin_variable_by_k(data, column, k):
    min_val = data[column].min()
    max_val = data[column].max()
    bins = np.linspace(min_val, max_val, k + 1)
    return bins

def bin_PDI(df, logger):
    k_PDI = binning_terrel_scott(df, 'PDI')
    bins_PDI = bin_variable_by_k(df, 'PDI', k_PDI)
    df['PDI_TS'] = pd.cut(df['PDI'], bins=bins_PDI, labels=False, include_lowest=True)
    bin_mapping = {i: 3 for i in range(4, 10)}
    df['PDI_CAT'] = df['PDI_TS'].replace(bin_mapping)

    logger.info(f"Numero di bin per 'PDI_CAT': {df['PDI_CAT'].nunique()}")
    logger.info("Conteggio degli elementi in ciascun bin di 'PDI_CAT':")
    logger.info(f"{df['PDI_CAT'].value_counts(sort=False)}")
    for bin_label in sorted(df['PDI_CAT'].dropna().unique()):
        if bin_label < len(bins_PDI) - 1:
            logger.info(f"Bin {bin_label}: {bins_PDI[bin_label]:.4f} - {bins_PDI[bin_label+1]:.4f}")

    return df
