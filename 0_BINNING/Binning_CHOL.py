import numpy as np
import pandas as pd
import math

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

def bin_CHOL(df, logger, h=3):
    k_CHOL = binning_max_min(df, 'CHOL', h=h)
    bins_CHOL = bin_variable_by_k(df, 'CHOL', k_CHOL)
    df['CHOL_CAT'] = pd.cut(df['CHOL'], bins=bins_CHOL, labels=False, include_lowest=True)
    
    logger.info(f"Numero di bin per 'CHOL_CAT': {df['CHOL_CAT'].nunique()}")
    logger.info("Conteggio degli elementi in ciascun bin di 'CHOL_CAT':")
    logger.info(f"{df['CHOL_CAT'].value_counts(sort=False)}")
    for i in range(len(bins_CHOL) - 1):
        logger.info(f"Bin {i}: da {bins_CHOL[i]:.4f} a {bins_CHOL[i+1]:.4f}")

    return df
