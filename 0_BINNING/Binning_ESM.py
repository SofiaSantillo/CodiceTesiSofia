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

def bin_ESM(df, logger, h=6):
    k_ESM = binning_max_min(df, 'ESM', h=h)
    bins_ESM = bin_variable_by_k(df, 'ESM', k_ESM)
    df['ESM_CAT'] = pd.cut(df['ESM'], bins=bins_ESM, labels=False, include_lowest=True)

    logger.info(f"Numero di bin per 'ESM': {len(bins_ESM) - 1}")
    logger.info("Conteggio degli elementi in ciascun bin di 'ESM_CAT':")
    logger.info(f"{df['ESM_CAT'].value_counts(sort=False)}")
    for i in range(len(bins_ESM) - 1):
        logger.info(f"Bin {i}: da {bins_ESM[i]:.4f} a {bins_ESM[i+1]:.4f}")

    return df