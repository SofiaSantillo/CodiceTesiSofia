import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

def plot_confronto_scalare(df1: pd.DataFrame, df2: pd.DataFrame, output_file="confronto_dataset.png"):
   
    comuni = set(df1.columns).intersection(set(df2.columns))
    numeriche = [col for col in comuni if np.issubdtype(df1[col].dtype, np.number) and np.issubdtype(df2[col].dtype, np.number)]
    
    if not numeriche:
        print("Non ci sono colonne numeriche in comune da confrontare.")
        return

    df1_plot = df1[numeriche]
    df2_plot = df2[numeriche]

    n_per_row = 4
    n_rows_grid = 2  
    ncols = n_per_row
    figures = []

    def plot_grid(data_list, titles, plot_func, tipo):
        n_plots = len(data_list)
        n_rows_needed = math.ceil(n_plots / n_per_row)
        fig, axes = plt.subplots(nrows=min(n_rows_needed, n_rows_grid), ncols=n_per_row, figsize=(5*n_per_row, 4*min(n_rows_needed, n_rows_grid)))
        axes = axes.flatten() if n_plots > 1 else [axes]
        
        for i, col in enumerate(data_list):
            plot_func(axes[i], col, tipo)
        
        for j in range(len(data_list), len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        return fig

    def plot_func(ax, col, tipo):
        if tipo == "hist":
            sns.histplot(df1_plot[col].dropna(), color='blue', label='data_1', stat='density', kde=False, ax=ax, alpha=0.5)
            sns.histplot(df2_plot[col].dropna(), color='red', label='ScaleUp', stat='density', kde=False, ax=ax, alpha=0.5)
            ax.set_title(f'{col} - Istogrammi')
        elif tipo == "box":
            data_box = [df1_plot[col].dropna(), df2_plot[col].dropna()]
            ax.boxplot(data_box, labels=['data_1', 'ScaleUp'])
            ax.set_title(f'{col} - Boxplot')
        elif tipo == "vals":
            vals1 = np.unique(df1_plot[col].dropna().values)
            vals2 = np.unique(df2_plot[col].dropna().values)
            ax.plot(vals1, label='data_1', color='blue')
            ax.plot(vals2, label='ScaleUp', color='red')
            ax.set_title(f'{col} - Valori unici ordinati')
        ax.legend()

    fig_hist = plot_grid(numeriche, numeriche, plot_func, "hist")
    fig_box = plot_grid(numeriche, numeriche, plot_func, "box")
    fig_vals = plot_grid(numeriche, numeriche, plot_func, "vals")

    fig_hist.savefig(f"hist_{output_file}", dpi=300)
    fig_box.savefig(f"box_{output_file}", dpi=300)
    fig_vals.savefig(f"vals_{output_file}", dpi=300)
    plt.close('all')
    print(f"Grafici salvati in 'hist_{output_file}', 'box_{output_file}', 'vals_{output_file}'")



df1 = pd.read_csv("_Data/data_1.csv")
df2 = pd.read_csv("_Data/dataset_ScaleUp.csv")

plot_confronto_scalare(df1, df2)
