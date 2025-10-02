import pandas as pd
import numpy as np

df = pd.read_csv('_Data/dataset_ScaleUp.csv')
numeric_cols = df.select_dtypes(include=np.number).columns

num_samples = len(df)
new_rows = []

for _ in range(num_samples):
    i, j = np.random.choice(num_samples, size=2, replace=False)
    alpha = np.random.rand()
    new_row = alpha * df.iloc[i][numeric_cols] + (1 - alpha) * df.iloc[j][numeric_cols]
    new_rows.append(new_row)

df_augmented = pd.DataFrame(new_rows, columns=numeric_cols)

df_augmented.loc[df_augmented["HSPC"] > 0, "ESM"] = 0
df_augmented.loc[df_augmented["ESM"] > 0, "HSPC"] = 0

df_augmented[numeric_cols] = df_augmented[numeric_cols].clip(lower=0)

df_final = pd.concat([df, df_augmented], ignore_index=True)
df_final.to_csv('_Data/dataset_ScaleUp_augmented_interp.csv', index=False)

print("Data augmentation con interpolazione completata. Nuovo file: _Data/dataset_ScaleUp_augmented_interp.csv")
