import pandas as pd


file_path = "Data_Droplet/validation_1.csv"  
df = pd.read_csv(file_path)

# Controllo che le colonne esistano
if not {"SIZE", "PDI"}.issubset(df.columns):
    raise ValueError("Il dataset non contiene le colonne SIZE e PDI")

# Calcolo della varianza
var_size = df["SIZE"].var()
var_pdi = df["PDI"].var()

print(f"Varianza SIZE: {var_size}")
print(f"Varianza PDI: {var_pdi}")