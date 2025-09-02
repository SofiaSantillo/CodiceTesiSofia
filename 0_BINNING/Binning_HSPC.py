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

def bin_HSPC(df, logger, h=5):
    k_HSPC = binning_max_min(df, 'HSPC', h=h)
    bins_HSPC = bin_variable_by_k(df, 'HSPC', k_HSPC)
    df['HSPC_CAT'] = pd.cut(df['HSPC'], bins=bins_HSPC, labels=False, include_lowest=True)

    logger.info(f"Numero di bin per 'HSPC': {len(bins_HSPC) - 1}")
    logger.info("Conteggio degli elementi in ciascun bin di 'HSPC_CAT':")
    logger.info(f"{df['HSPC_CAT'].value_counts(sort=False)}")
    for i in range(len(bins_HSPC) - 1):
        logger.info(f"Bin {i}: da {bins_HSPC[i]:.4f} a {bins_HSPC[i+1]:.4f}")

    return df