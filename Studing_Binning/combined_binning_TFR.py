import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def binning_sqrt(data, column):
    n = data[column].shape[0]
    k = math.ceil(math.sqrt(n))
    return k

def binning_max_min(data, column, h):
    max_val = data[column].max()  
    min_val = data[column].min() 
    k = math.ceil((max_val - min_val) / h) 
    return k

def binning_rice_rule(data, column):
    n = data[column].shape[0] 
    k = math.ceil(2 * (n ** (1/3))) 
    return k

def binning_terrel_scott(data, column):
    n = data[column].shape[0]
    k = math.ceil((2 * n) ** (1/3))
    return k

# Funzione per calcolare h = 2 * (IQR(x) / n^(1/3))
def binning_freedman_diaconis(data, column):
    n = data[column].shape[0]  
    iqr = data[column].quantile(0.75) - data[column].quantile(0.25) 
    h = 2 * (iqr / (n ** (1/3)))  # Calcolo di h come 2 * (IQR / radice cubica di n)
    max_val = data[column].max() 
    min_val = data[column].min() 
    k = math.ceil((max_val - min_val) / h)
    return k

# Funzione per creare i bin sulla base di k e restituire i bin
def bin_variable_by_k(data, column, k):
    min_val = data[column].min()
    max_val = data[column].max()
    bins = np.linspace(min_val, max_val, k + 1)
    return pd.cut(data[column], bins=bins, include_lowest=True)


if __name__ == "__main__":
    file_path = "Data_DAG/Nodi_DAG1.2.csv"

    # Carica il dataset
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Formato file non supportato (usa .csv o .xlsx)")

   
    k_bins_sqrt = binning_sqrt(df, 'TFR')
    df['TFR_binned_sqrt'] = bin_variable_by_k(df, 'TFR', k_bins_sqrt)


    k_bins_distribution = binning_max_min(df, 'TFR', h=0.5)
    df['TFR_binned_distribution'] = bin_variable_by_k(df, 'TFR', k_bins_distribution)

  
    k_bins_rice = binning_rice_rule(df, 'TFR')
    df['TFR_binned_rice'] = bin_variable_by_k(df, 'TFR', k_bins_rice)


    k_bins_ts = binning_terrel_scott(df, 'TFR')
    df['TFR_binned_ts'] = bin_variable_by_k(df, 'TFR', k_bins_ts)


    k_bins_fd = binning_freedman_diaconis(df, 'TFR')
    df['TFR_binned_fd'] = bin_variable_by_k(df, 'TFR', k_bins_fd)
    

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # Due righe, tre colonne

    # Primo istogramma (basato su √n)
    df['TFR_binned_sqrt'].value_counts().sort_index().plot(kind='bar', ax=axes[0, 0], color='orange', edgecolor='black')
    axes[0, 0].set_title(f"Istogramma binned di TFR (k=⌈√n⌉)")
    axes[0, 0].set_xlabel("Intervalli")
    axes[0, 0].set_ylabel("Frequenza")
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Secondo istogramma (basato sulla distribuzione dei dati)
    df['TFR_binned_distribution'].value_counts().sort_index().plot(kind='bar', ax=axes[0, 1], color='green', edgecolor='black')
    axes[0, 1].set_title(f"Istogramma binned di TFR (k=|max-min/h|)")
    axes[0, 1].set_xlabel("Intervalli")
    axes[0, 1].set_ylabel("Frequenza")
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Terzo istogramma (basato sulla regola del riso)
    df['TFR_binned_rice'].value_counts().sort_index().plot(kind='bar', ax=axes[0, 2], color='yellow', edgecolor='black')
    axes[0, 2].set_title(f"Istogramma binned di TFR (Rice Rule)")
    axes[0, 2].set_xlabel("Intervalli")
    axes[0, 2].set_ylabel("Frequenza")
    axes[0, 2].tick_params(axis='x', rotation=45)

    # Quarto istogramma (basato sulla regola Terrell-Scott)
    df['TFR_binned_ts'].value_counts().sort_index().plot(kind='bar', ax=axes[1, 0], color='red', edgecolor='black')
    axes[1, 0].set_title(f"Istogramma binned di TFR (Terrell-Scott)")
    axes[1, 0].set_xlabel("Intervalli")
    axes[1, 0].set_ylabel("Frequenza")
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Quinto istogramma (basato sulla regola di Freedman-Diaconis)
    df['TFR_binned_fd'].value_counts().sort_index().plot(kind='bar', ax=axes[1, 1], color='blue', edgecolor='black')
    axes[1, 1].set_title(f"Istogramma binned di TFR (Freedman-Diaconis)")
    axes[1, 1].set_xlabel("Intervalli")
    axes[1, 1].set_ylabel("Frequenza")
    axes[1, 1].tick_params(axis='x', rotation=45)


    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig("Studing_Binning/combined_histogram_TFR.png")
    plt.close()
