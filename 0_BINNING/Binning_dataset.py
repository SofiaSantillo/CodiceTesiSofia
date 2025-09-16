import pandas as pd
import os
import logging
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
import math
from sklearn.preprocessing import OneHotEncoder

from Binning_FRR import bin_FRR
from Binning_SIZE import bin_SIZE
from Binning_PDI import bin_PDI
from Binning_TFR import bin_TFR
from Binning_ESM import bin_ESM
from Binning_HSPC import bin_HSPC
from Binning_CHOL import bin_CHOL
from Binning_PEG import bin_PEG

log_path = '0_BINNING/_logs'
dataset_path = '_Data/data_1.csv'
plot_path = "0_BINNING/Plot_binning"

if not os.path.exists(log_path):
    os.makedirs(log_path)

def remove_cat_suffix(df):
    df.columns = [col.replace('_CAT', '') for col in df.columns]
    return df

def setup_logger(name, log_file, level=logging.INFO):
    """Crea e restituisce un logger con handler separato"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.FileHandler(log_file, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

def creation_dataset(log_path, dataset_path):

    logger_FRR = setup_logger('FRR', os.path.join(log_path, f'Binning_FRR.log'))
    logger_SIZE = setup_logger('SIZE', os.path.join(log_path, f'Binning_SIZE.log'))
    logger_PDI = setup_logger('PDI', os.path.join(log_path, f'Binning_PDI.log'))
    logger_TFR = setup_logger('TFR', os.path.join(log_path, f'Binning_TFR.log'))
    logger_ESM = setup_logger('ESM', os.path.join(log_path, f'Binning_ESM.log'))
    logger_HSPC = setup_logger('HSPC', os.path.join(log_path, f'Binning_HSPC.log'))
    logger_CHOL = setup_logger('CHOL', os.path.join(log_path, f'Binning_CHOL.log'))
    logger_PEG = setup_logger('PEG', os.path.join(log_path, f'Binning_PEG.log'))

    df = pd.read_csv(dataset_path)

    df_cleaned = df.dropna().copy()

    df_cleaned = bin_FRR(df_cleaned, logger_FRR)
    df_cleaned = bin_SIZE(df_cleaned, logger_SIZE)
    df_cleaned = bin_PDI(df_cleaned, logger_PDI)
    df_cleaned = bin_TFR(df_cleaned, logger_TFR)
    df_cleaned = bin_ESM(df_cleaned, logger_ESM)
    df_cleaned = bin_HSPC(df_cleaned, logger_HSPC)
    df_cleaned = bin_CHOL(df_cleaned, logger_CHOL)
    df_cleaned = bin_PEG(df_cleaned, logger_PEG)

    df_new = df_cleaned.drop(columns=[
        "PDI", "SIZE", "FRR", "TFR", "PDI_TS",
        "ESM", "HSPC", "CHOL", "CHOL_TS", "PEG"
    ], errors="ignore")

    df_new = remove_cat_suffix(df_new)

    cat_cols = df_new.select_dtypes(include=['object', 'category']).columns.tolist()

    for col in df_new.select_dtypes(include=['object']).columns:
        df_new[col], _ = pd.factorize(df_new[col])

    output_file = os.path.join('_Data', 'data_1_Binning.csv')
    df_new.to_csv(output_file, index=False)

    return df_new


if __name__ == "__main__":
    df = creation_dataset(log_path, dataset_path)
