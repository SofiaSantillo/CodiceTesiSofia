import pandas as pd

input_file = "Extraction_dataset/DataSet_ScaleUp.xlsx"
output_file = "Data_Droplet/dataset_ScaleUp.csv"

df = pd.read_excel(input_file)
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

df.rename(columns=rename_dict, inplace=True)
df.to_csv(output_file, index=False)

print(f"File CSV salvato come: {output_file}")