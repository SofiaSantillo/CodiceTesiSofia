import numpy as np
import pandas as pd
import math
import logging

def binning_max_min(data, column, h):
    max_val = data[column].max() 
    min_val = data[column].min() 
    k = math.ceil((max_val - min_val) / h)  
    return k

def bin_variable_by_k(data, column, k):
    min_val = data[column].min()
    max_val = data[column].max()
    bins = np.linspace(min_val, max_val, k + 1)
    return bins

def bin_FRR(df, logger, h=15):
    k_FRR = binning_max_min(df, 'FRR', h=h)
    bins_FRR = bin_variable_by_k(df, 'FRR', k_FRR)
    df['FRR_CAT'] = pd.cut(df['FRR'], bins=bins_FRR, labels=False, include_lowest=True)

    logger.info(f"Numero di bin per 'FRR': {len(bins_FRR) - 1}")
    logger.info("Conteggio degli elementi in ciascun bin di 'FRR_CAT':")
    logger.info(f"{df['FRR_CAT'].value_counts(sort=False)}")
    for i in range(len(bins_FRR) - 1):
        logger.info(f"Bin {i}: da {bins_FRR[i]:.4f} a {bins_FRR[i+1]:.4f}")

    return df