import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_distributions(df, output_folder="0_BINNING/Plot_binning"):
    """
    Plotta le distribuzioni di tutte le colonne di un dataframe e salva i grafici in PNG separati.

    Args:
        df (pd.DataFrame): Dataset
        output_folder (str): Cartella dove salvare i grafici
    """
    os.makedirs(output_folder, exist_ok=True)

    for col in df.columns:
        plt.figure(figsize=(8, 6))

        if pd.api.types.is_numeric_dtype(df[col]):
            sns.histplot(df[col].dropna(), kde=True, bins=30)
            plt.title(f'Distribution of {col} (numeric)')
        else:
            sns.countplot(x=col, data=df)
            plt.title(f'Distribution of {col} (categorical)')
            plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{col}_distribution.png"))
        plt.close()

if __name__ == "__main__":

    df = pd.read_csv("_Data/data_1.csv")  
    plot_distributions(df)
