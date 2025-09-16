import pandas as pd
from sklearn.utils import shuffle

# --- Carica i due dataset ---
df1 = pd.read_csv("_Data/seed_non_ordinato.csv")
df2 = pd.read_csv("Data/dataset_ScaleUp.csv")

n1 = len(df1)
n2 = len(df2)

# --- Unisci e mescola solo le righe (senza modificarle) ---
all_data = pd.concat([df1, df2], ignore_index=True)
all_data = shuffle(all_data, random_state=42).reset_index(drop=True)

# --- Ridistribuisci mantenendo le dimensioni originali ---
mixed1 = all_data.iloc[:n1].copy()
mixed2 = all_data.iloc[n1:n1+n2].copy()

# --- Salva i nuovi dataset ---
mixed1.to_csv("_Data/dataset_ScaleUp_mixed_train.csv", index=False)
mixed2.to_csv("_Data/dataset_ScaleUp_mixed_validation.csv", index=False)

print("Dataset mescolati correttamente: righe intatte, solo ordine cambiato!")
