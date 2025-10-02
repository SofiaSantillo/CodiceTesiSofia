import pandas as pd
import numpy as np

dataset1 = pd.read_csv('_Data/data_1.csv')
dataset2 = pd.read_csv('_Data/dataset_ScaleUp.csv')

clusters = [(0,3), (3,6), (6,12), (12,15), (15,25)]
cluster_ranges = {}

epsilon = 1e-3
max_factor = 25.0

for low, high in clusters:
    vals1 = dataset1['ESM'][(dataset1['ESM'] >= low) & (dataset1['ESM'] < high)].values
    vals2 = dataset2['ESM'][(dataset2['ESM'] >= low) & (dataset2['ESM'] < high)].values

    if len(vals1) == 0 or len(vals2) == 0:
        continue

    cdf1 = np.linspace(0,1,len(vals1))
    cdf2 = np.linspace(0,1,len(vals2))

    min1, max1 = np.min(vals1), np.max(vals1)
    min2, max2 = np.min(vals2), np.max(vals2)

    min2 = max(min2, epsilon)
    max2 = max(max2, epsilon)

    s_min = max(min1 / max2, 0.01)
    s_max = min(max1 / min2, max_factor)

    cluster_ranges[f"{low}-{high}"] = (s_min, s_max)

print("📊 Range fattori di scala ESM (CDF approach):")
for cluster, (smin, smax) in cluster_ranges.items():
    print(f"{cluster}: [{smin:.3f}, {smax:.3f}]")
