import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset1 = pd.read_csv('_Data/data_1.csv')
dataset2 = pd.read_csv('_Data/dataset_ScaleUp.csv')

clusters = [
    (0, 3),
    (3, 25),
    (25, dataset2['TFR'].max())
]

cluster_ranges = {}

for low, high in clusters:
    cluster_values = dataset2['TFR'][(dataset2['TFR'] >= low) & (dataset2['TFR'] < high)].values
    
    if len(cluster_values) == 0:
        continue
    
    p_low, p_high = 5, 95
    d1_low, d1_high = np.percentile(dataset1['TFR'], [p_low, p_high])
    cluster_low, cluster_high = np.percentile(cluster_values, [p_low, p_high])
    
    s_min = d1_low / cluster_high
    s_max = d1_high / cluster_low
    cluster_ranges[f"{low}-{high}"] = (s_min, s_max)

for cluster, (smin, smax) in cluster_ranges.items():
    print(f"Cluster {cluster}: range fattori di scala = [{smin:.3f}, {smax:.3f}]")

plt.figure(figsize=(10,5))
bins = np.linspace(0, max(dataset1['TFR'].max(), dataset2['TFR'].max()), 50)

plt.hist(dataset1['TFR'], bins=bins, alpha=0.5, label='Dataset1', color='blue', density=True)
plt.hist(dataset2['TFR'], bins=bins, alpha=0.5, label='Dataset2', color='red', density=True)

tfr_scaled = dataset2['TFR'].copy()
for cluster, (smin, smax) in cluster_ranges.items():
    low, high = map(float, cluster.split('-'))
    mask = (dataset2['TFR'] >= low) & (dataset2['TFR'] < high)
    factor = (smin + smax) / 2
    tfr_scaled[mask] = dataset2['TFR'][mask] * factor



print(tfr_scaled)