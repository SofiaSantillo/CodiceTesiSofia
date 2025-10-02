import pandas as pd
import numpy as np

dataset1 = pd.read_csv('_Data/data_1.csv')
dataset2 = pd.read_csv('_Data/dataset_ScaleUp.csv')

clusters = [(0,5), (5,10), (10,20), (20,50)]

cluster_ranges_frr = {}
epsilon = 1e-6
max_factor = 20

for low, high in clusters:
    vals1 = dataset1['FRR'][(dataset1['FRR'] >= low) & (dataset1['FRR'] < high)].values
    vals2 = dataset2['FRR'][(dataset2['FRR'] >= low) & (dataset2['FRR'] < high)].values
    
    if len(vals1) == 0 or len(vals2) == 0:
        continue
    
    p_low, p_high = np.percentile(vals1, [5,95])
    c_low, c_high = np.percentile(vals2, [1,99])
    
    c_low = max(c_low, epsilon)
    c_high = max(c_high, epsilon)
    
    s_min = max(p_low / c_high, 0.01)
    s_max = min(p_high / c_low, max_factor)
    
    cluster_ranges_frr[f"{low}-{high}"] = (s_min, s_max)

print("📊 Range fattori di scala per FRR:")
for cluster, (smin, smax) in cluster_ranges_frr.items():
    print(f"{cluster}: [{smin:.3f}, {smax:.3f}]")
