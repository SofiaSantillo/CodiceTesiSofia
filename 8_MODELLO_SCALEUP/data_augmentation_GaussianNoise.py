import pandas as pd
import numpy as np

df = pd.read_csv('_Data/dataset_ScaleUp.csv')
numeric_cols = df.select_dtypes(include=np.number).columns

sigma_fraction = 0.05
sigma = df[numeric_cols].std() * sigma_fraction

df_augmented = df.copy()
df_augmented[numeric_cols] = df[numeric_cols] + np.random.normal(0, sigma.values, size=df[numeric_cols].shape)

df_augmented.loc[df_augmented["HSPC"] > 0, "ESM"] = 0
df_augmented.loc[df_augmented["ESM"] > 0, "HSPC"] = 0

df_augmented[numeric_cols] = df_augmented[numeric_cols].clip(lower=0)

df_final = pd.concat([df, df_augmented], ignore_index=True)
df_final.to_csv('_Data/dataset_ScaleUp_augmented_GN.csv', index=False)

print("Data augmentation completata. Nuovo file: _Data/dataset_ScaleUp_augmented_GN.csv")
