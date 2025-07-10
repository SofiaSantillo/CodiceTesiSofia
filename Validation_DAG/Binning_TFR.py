import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
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

def bin_TFR(df_cleaned, logger):

    # Binning TFR con metodo max-min
    k_TFR = binning_max_min(df_cleaned, 'TFR', h=0.5)
    bins_TFR = bin_variable_by_k(df_cleaned, 'TFR', k_TFR)
    df_cleaned['TFR_CAT'] = pd.cut(df_cleaned['TFR'], bins=bins_TFR, labels=False, include_lowest=True)

    logger.info(f"\nNumero di bin per 'TFR': {len(bins_TFR) - 1}\n")
    logger.info("\nConteggio degli elementi in ciascun bin di 'TFR_CAT':\n")
    logger.info(f"{df_cleaned['TFR_CAT'].value_counts(sort=False)}\n")
    logger.info("\nIntervalli dei bin TFR:\n")
    for i in range(len(bins_TFR) - 1):
        logger.info(f"Bin {i}: da {bins_TFR[i]:.4f} a {bins_TFR[i+1]:.4f}\n")

    return df_cleaned
