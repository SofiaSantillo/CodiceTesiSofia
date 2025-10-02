import pandas as pd
import numpy as np

dataset1 = pd.read_csv("_Data/data_1.csv")
dataset2 = pd.read_csv("_Data/dataset_ScaleUp.csv")

clusters = [(0,5),(5,10),(10,15),(15,22)]
cluster_ranges_hspc = {}
epsilon = 1e-3
max_factor = 25

vals2_nonzero = dataset2["HSPC"][dataset2["HSPC"] > epsilon].values

for low, high in clusters:
    vals1 = dataset1["HSPC"][(dataset1["HSPC"] >= low) & (dataset1["HSPC"] < high)].values
    if len(vals1) == 0 or len(vals2_nonzero) == 0:
        continue
    
    p1_low, p1_high = np.percentile(vals1, [5,95])
    
    val2 = np.median(vals2_nonzero)
    
    s_min = max(p1_low / (val2 + epsilon), 0.01)
    s_max = min(p1_high / (val2 + epsilon), max_factor)
    
    cluster_ranges_hspc[f"{low}-{high}"] = (s_min, s_max)

print("📊 Range fattori di scala per HSPC (solo valori ≠ 0 in dataset2):")
for cluster, (smin, smax) in cluster_ranges_hspc.items():
    print(f"Cluster {cluster}: [{smin:.3f}, {smax:.3f}]")

print("\nNota: i valori 0 in dataset2 non sono scalabili e rimangono invariati.")
