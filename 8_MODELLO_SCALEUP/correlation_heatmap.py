import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("_Data/dataset_ScaleUp.csv")

numeric_cols = dataset.select_dtypes(include="number").columns
dataset_numeric = dataset[numeric_cols]

corr_matrix = dataset_numeric.corr()
print(corr_matrix)

plt.figure(figsize=(10, 8))

sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap - Dataset ScaleUp")
plt.show()