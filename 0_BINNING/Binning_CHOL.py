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

def bin_CHOL(df, logger):
    k_CHOL = binning_terrel_scott(df, 'CHOL')
    bins_CHOL = bin_variable_by_k(df, 'CHOL', k_CHOL)
    df['CHOL_TS'] = pd.cut(df['CHOL'], bins=bins_CHOL, labels=False, include_lowest=True)
    bin_mapping = {i: 3 for i in range(6, 9)}
    df['CHOL_CAT'] = df['CHOL_TS'].replace(bin_mapping)

    logger.info(f"Numero di bin per 'CHOL_CAT': {df['CHOL_CAT'].nunique()}")
    logger.info("Conteggio degli elementi in ciascun bin di 'CHOL_CAT':")
    logger.info(f"{df['CHOL_CAT'].value_counts(sort=False)}")
    for bin_label in sorted(df['CHOL_CAT'].dropna().unique()):
        if bin_label < len(bins_CHOL) - 1:
            logger.info(f"Bin {bin_label}: {bins_CHOL[bin_label]:.4f} - {bins_CHOL[bin_label+1]:.4f}")

    return df
