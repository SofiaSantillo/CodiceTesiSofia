import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Carica il file CSV originale
df = pd.read_csv("Data_DLS/data_DLS.csv")

# Crea una directory per i grafici se non esiste
output_dir = "Plots"
os.makedirs(output_dir, exist_ok=True)

# Forza la conversione delle colonne a tipo numerico, impostando errori a NaN
df = df.apply(pd.to_numeric, errors='coerce')

# Rimozione outlier tramite IQR su tutte le colonne numeriche
for col_name in df.select_dtypes(include=[float, int]).columns:
    col = df[col_name].dropna()

    if col.nunique() <= 1:
        continue

    Q1 = col.quantile(0.25)
    Q3 = col.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    col_clean = col[(col >= lower_bound) & (col <= upper_bound)]

    # Grafico prima e dopo l'IQR
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(col, kde=True, bins=30, ax=axes[0])
    axes[0].set_title(f"{col_name} - Before IQR")
    sns.histplot(col_clean, kde=True, bins=30, ax=axes[1])
    axes[1].set_title(f"{col_name} - After IQR")

    # Salva grafico
    plot_path = os.path.join(output_dir, f"{col_name}_iqr_comparison.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    # Applica la maschera per rimuovere outlier
    df[col_name] = df[col_name].apply(lambda x: x if pd.isna(x) or (lower_bound <= x <= upper_bound) else pd.NA)

    print(f"Processed column: {col_name}, plot saved to {plot_path}")

# Rimuovi le righe che contengono valori mancanti in colonne diverse dalla prima
columns_to_check = df.columns[1:]  # tutte tranne la prima
df_final = df.dropna(subset=columns_to_check)

# Salva il DataFrame finale in un nuovo CSV
df_final.to_csv("Data_DLS/data_DLS_cleaned.csv", index=False)

print("Elaborazione completa. File finale salvato in data_DLS_final_cleaned.csv")
