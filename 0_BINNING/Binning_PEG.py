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

def bin_PEG(df, logger, h=1.5):
    k_PEG = binning_max_min(df, 'PEG', h=h)
    bins_PEG = bin_variable_by_k(df, 'PEG', k_PEG)
    df['PEG_CAT'] = pd.cut(df['PEG'], bins=bins_PEG, labels=False, include_lowest=True)

    logger.info(f"Numero di bin per 'PEG': {len(bins_PEG) - 1}")
    logger.info("Conteggio degli elementi in ciascun bin di 'PEG_CAT':")
    logger.info(f"{df['PEG_CAT'].value_counts(sort=False)}")
    for i in range(len(bins_PEG) - 1):
        logger.info(f"Bin {i}: da {bins_PEG[i]:.4f} a {bins_PEG[i+1]:.4f}")

    return df