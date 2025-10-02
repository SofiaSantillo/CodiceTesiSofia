import pandas as pd
import numpy as np

dataset1 = pd.read_csv("_Data/data_1.csv")
dataset2 = pd.read_csv("_Data/dataset_ScaleUp.csv")

clusters = [(0,5),(5,10),(10,15),(15,20)]
cluster_ranges = {}
epsilon = 1e-3
max_factor = 20  

for low, high in clusters:
    vals1 = dataset1["CHOL"][(dataset1["CHOL"] >= low) & (dataset1["CHOL"] < high)].values
    vals2 = dataset2["CHOL"][(dataset2["CHOL"] >= low) & (dataset2["CHOL"] < high)].values
    
    if len(vals1) == 0 or len(vals2) == 0:
        continue
    
    p1_low, p1_high = np.percentile(vals1, [5,95])
    p2_low, p2_high = np.percentile(vals2, [1,99])
    
    p2_low = max(p2_low, epsilon)
    p2_high = max(p2_high, epsilon)
    
    s_min = max(p1_low / p2_high, 0.01)
    s_max = min(p1_high / p2_low, max_factor)
    
    cluster_ranges[f"{low}-{high}"] = (s_min, s_max)

print("📊 Range fattori di scala per CHOL:")
for cluster, (smin, smax) in cluster_ranges.items():
    print(f"Cluster {cluster}: [{smin:.3f}, {smax:.3f}]")
