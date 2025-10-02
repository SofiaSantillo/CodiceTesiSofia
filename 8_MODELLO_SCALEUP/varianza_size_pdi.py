import pandas as pd


file_path = "_Data/dataset_ScaleUp.csv"  
df = pd.read_csv(file_path)

if not {"SIZE", "PDI"}.issubset(df.columns):
    raise ValueError("Il dataset non contiene le colonne SIZE e PDI")

var_size_0 = df["SIZE"].var(ddof=0)
var_pdi_0 = df["PDI"].var(ddof=0)

var_size = df["SIZE"].var(ddof=1)
var_pdi = df["PDI"].var(ddof=1)

print(f"Varianza SIZE: {var_size_0}")
print(f"Varianza PDI: {var_pdi_0}")
print(f"Varianza SIZE: {var_size}")
print(f"Varianza PDI: {var_pdi}")