import numpy as np
import pandas as pd
from itertools import combinations

log_file = "3_D-SEPARATION/_logs/paths_C272_M473.log"
data_file = "Data_Droplet/seed_Binning_ordinato.csv" 


def parse_paths_from_log(file_path):
    paths = []
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("Percorso:"):
                start_idx = line.find("[")
                end_idx = line.find("]") + 1
                if start_idx != -1 and end_idx != -1:
                    path_str = line[start_idx:end_idx]
                    path = eval(path_str)
                    paths.append(path)
    return paths

def entropy(arr):
    vals, counts = np.unique(arr, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))

def joint_entropy(arrays):
    if isinstance(arrays, np.ndarray):
        if arrays.ndim == 1:
            arrays = [arrays]
        elif arrays.ndim == 2:
            arrays = [arrays[:, i] for i in range(arrays.shape[1])]
    arrays = [np.asarray(a).ravel() for a in arrays]
    df = pd.DataFrame(arrays).T
    counts = df.value_counts(normalize=True)
    return -np.sum(counts * np.log2(counts))

def conditional_mutual_information(x, y, z=None):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if z is not None and len(z) > 0:
        z = [np.asarray(col).ravel() for col in z]  
        XZ = np.column_stack([x] + z)
        YZ = np.column_stack([y] + z)
        XYZ = np.column_stack([x, y] + z)
        Z_only = np.column_stack(z)
        return joint_entropy(XZ) + joint_entropy(YZ) - joint_entropy(XYZ) - joint_entropy(Z_only)
    else:
        XY = np.column_stack([x, y])
        return joint_entropy(x) + joint_entropy(y) - joint_entropy(XY)


def compute_cmi_paths(paths, df, x_col, y_col, epsilon=0.05):
    results = []

    for path in paths:
        if path[0] != x_col or path[-1] != y_col:
            path = path[::-1]

        intermediates = path[1:-1]

        for r in range(len(intermediates) + 1):
            for cond_set in combinations(intermediates, r):
                if cond_set:
             
                    combined = pd.concat([df[x_col], df[y_col]] + [df[col] for col in cond_set], axis=1).dropna()
                    x_vals_clean = combined[x_col].values  
                    y_vals_clean = combined[y_col].values  
                    z_vals_clean = [combined[col].values for col in cond_set]  
   
                    cmi = conditional_mutual_information(x_vals_clean, y_vals_clean, z_vals_clean)
                else:
                    combined = pd.concat([df[x_col], df[y_col]], axis=1).dropna()
                    x_vals_clean = combined[x_col].values
                    y_vals_clean = combined[y_col].values
                    cmi = conditional_mutual_information(x_vals_clean, y_vals_clean, None)

                results.append({
                    "path": path,
                    "conditioned_on": cond_set if cond_set else None,
                    "cmi": cmi,
                    "d_separated": cmi < epsilon
                })

    return pd.DataFrame(results)


if __name__ == "__main__":
    df = pd.read_csv(data_file)
    paths = parse_paths_from_log(log_file)

    cmi_df = compute_cmi_paths(paths, df, x_col="ESM", y_col="HSPC")
    print(cmi_df)
    cmi_df.to_csv("3_D-SEPARATION/_logs/cmi_C272_M473.log", index=False)
