import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("_Output", exist_ok=True)

dataset1 = pd.read_csv('_Data/data_1.csv')
dataset2 = pd.read_csv('_Data/dataset_ScaleUp.csv')

clusters = [(0,3),(3,6),(6,12),(12,15)]
cluster_ranges = {}
epsilon = 1e-3
max_factor = 20

for low, high in clusters:
    vals2 = dataset2['PEG'][(dataset2['PEG'] >= low) & (dataset2['PEG'] < high)].values
    vals1 = dataset1['PEG'][(dataset1['PEG'] >= low) & (dataset1['PEG'] < high)].values
    if len(vals2) == 0 or len(vals1) == 0:
        continue
    p_low, p_high = np.percentile(vals1, [5,95])
    c_low, c_high = np.percentile(vals2, [1,99])
    c_low = max(c_low, epsilon)
    c_high = max(c_high, epsilon)
    s_min = max(p_low / c_high, 0.01)
    s_max = min(p_high / c_low, max_factor)
    cluster_ranges[f"{low}-{high}"] = (s_min, s_max)

dataset2_scaled = dataset2.copy()
for low, high in clusters:
    mask = (dataset2['PEG'] >= low) & (dataset2['PEG'] < high)
    s_min, s_max = cluster_ranges.get(f"{low}-{high}", (1,1))
    factor = (s_min + s_max) / 2
    dataset2_scaled.loc[mask, 'PEG'] *= factor

