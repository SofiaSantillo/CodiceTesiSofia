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
    h = 2 * (iqr / (n ** (1/3)))  
    max_val = data[column].max() 
    min_val = data[column].min()  
    k = math.ceil((max_val - min_val) / h)
    return k

def binning_sturges(data, column):
    n = data[column].shape[0]
    k = math.ceil(math.log2(n)) + 1
    return k

# Funzione per creare i bin sulla base di k e restituire i bin
def bin_variable_by_k(data, column, k):
    min_val = data[column].min()
    max_val = data[column].max()
    bins = np.linspace(min_val, max_val, k + 1)
    return pd.cut(data[column], bins=bins, include_lowest=True)


if __name__ == "__main__":
    file_path = "_Data/data_1.csv"

 
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Formato file non supportato (usa .csv o .xlsx)")


    k_bins_sqrt = binning_sqrt(df, 'SIZE')
    df['SIZE_binned_sqrt'] = bin_variable_by_k(df, 'SIZE', k_bins_sqrt)


    k_bins_distribution = binning_max_min(df, 'SIZE', h=100)
    df['SIZE_binned_distribution'] = bin_variable_by_k(df, 'SIZE', k_bins_distribution)

   
    k_bins_rice = binning_rice_rule(df, 'SIZE')
    df['SIZE_binned_rice'] = bin_variable_by_k(df, 'SIZE', k_bins_rice)

   
    k_bins_ts = binning_terrel_scott(df, 'SIZE')
    df['SIZE_binned_ts'] = bin_variable_by_k(df, 'SIZE', k_bins_ts)


    k_bins_fd = binning_freedman_diaconis(df, 'SIZE')
    df['SIZE_binned_fd'] = bin_variable_by_k(df, 'SIZE', k_bins_fd)

    k_bins_sturges = binning_sturges(df, 'SIZE')
    df['SIZE_binned_sturges'] = bin_variable_by_k(df, 'SIZE', k_bins_sturges)
    

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))  

    # Primo istogramma (basato su √n)
    df['SIZE_binned_sqrt'].value_counts().sort_index().plot(kind='bar', ax=axes[0, 0], color='orange', edgecolor='black')
    axes[0, 0].set_title(f"Istogramma binned di SIZE (k=⌈√n⌉)")
    axes[0, 0].set_xlabel("Intervalli")
    axes[0, 0].set_ylabel("Frequenza")
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Secondo istogramma (basato sulla distribuzione dei dati)
    df['SIZE_binned_distribution'].value_counts().sort_index().plot(kind='bar', ax=axes[0, 1], color='green', edgecolor='black')
    axes[0, 1].set_title(f"Istogramma binned di SIZE (k=|max-min/h|)")
    axes[0, 1].set_xlabel("Intervalli")
    axes[0, 1].set_ylabel("Frequenza")
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Terzo istogramma (basato sulla regola del riso)
    df['SIZE_binned_rice'].value_counts().sort_index().plot(kind='bar', ax=axes[0, 2], color='yellow', edgecolor='black')
    axes[0, 2].set_title(f"Istogramma binned di SIZE (Rice Rule)")
    axes[0, 2].set_xlabel("Intervalli")
    axes[0, 2].set_ylabel("Frequenza")
    axes[0, 2].tick_params(axis='x', rotation=45)

    # Quarto istogramma (basato sulla regola Terrell-Scott)
    df['SIZE_binned_ts'].value_counts().sort_index().plot(kind='bar', ax=axes[1, 0], color='red', edgecolor='black')
    axes[1, 0].set_title(f"Istogramma binned di SIZE (Terrell-Scott)")
    axes[1, 0].set_xlabel("Intervalli")
    axes[1, 0].set_ylabel("Frequenza")
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Quinto istogramma (basato sulla regola di Freedman-Diaconis)
    df['SIZE_binned_fd'].value_counts().sort_index().plot(kind='bar', ax=axes[1, 1], color='blue', edgecolor='black')
    axes[1, 1].set_title(f"Istogramma binned di SIZE (Freedman-Diaconis)")
    axes[1, 1].set_xlabel("Intervalli")
    axes[1, 1].set_ylabel("Frequenza")
    axes[1, 1].tick_params(axis='x', rotation=45)

    # Sesto istogramma (basato sulla regola di Freedman-Diaconis)
    df['SIZE_binned_fd'].value_counts().sort_index().plot(kind='bar', ax=axes[1, 2], color='violet', edgecolor='black')
    axes[1, 2].set_title(f"Istogramma binned di SIZE (Sturges)")
    axes[1, 2].set_xlabel("Intervalli")
    axes[1, 2].set_ylabel("Frequenza")
    axes[1, 2].tick_params(axis='x', rotation=45)


    plt.tight_layout()
    plt.savefig("0_BINNING/Plot_binning/combined_histogram_SIZE.png")
    plt.close()
