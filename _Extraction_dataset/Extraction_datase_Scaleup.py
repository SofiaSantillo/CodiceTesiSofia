import pandas as pd

# Percorso del file Excel di input
input_file = "Extraction_dataset/DataSet_ScaleUp.xlsx"

# Percorso del file CSV di output
output_file = "Data_Droplet/dataset_ScaleUp.csv"

# Legge il file Excel
df = pd.read_excel(input_file)
# Dizionario di rinomina colonne
rename_dict = {
    "Main Lipid": "ML",
    "Chip": "CHIP",
    "ESM concentration (mg/mL)": "ESM",
    "HSPC concentration mg/mL": "HSPC",
    "Cholesterol concentration (mg/mL)": "CHOL",
    "DSPE-PEG2000 concentration mg/mL": "PEG",
    "TFR mL/min": "TFR",
    "FRR": "FRR",
    "Aqueous medium": "AQUEOUS",
    "Formation": "OUTPUT",
    "Size (nm)": "SIZE",
    "PDI": "PDI"
}

# Rinomina le colonne
df.rename(columns=rename_dict, inplace=True)

# Salva il dataset in formato CSV
df.to_csv(output_file, index=False)

print(f"File CSV salvato come: {output_file}")