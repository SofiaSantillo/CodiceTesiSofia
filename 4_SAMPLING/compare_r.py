import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from scipy.stats import ks_2samp
import glob
import os
import warnings

# ------------------------------ Setup ------------------------------
real_csv_size = "4_SAMPLING/_csv/real_r_SIZE.csv"
real_csv_pdi = "4_SAMPLING/_csv/real_r_PDI.csv"
sim_folder = "4_SAMPLING/_csv"
log_folder = "4_SAMPLING/_logs"
os.makedirs(log_folder, exist_ok=True)
log_file = os.path.join(log_folder, "comparison_sampling.log")

# ------------------------------ Funzioni ------------------------------
def compute_metrics(sim_file, real_file, outcome_name):
    df_sim = pd.read_csv(sim_file).dropna(subset=[outcome_name])
    df_real = pd.read_csv(real_file).dropna(subset=[outcome_name])
    
    df_sim = df_sim.rename(columns={outcome_name: f"{outcome_name}_sim"})
    df_real = df_real.rename(columns={outcome_name: f"{outcome_name}_real"})
    
    # Merge sulle feature comuni se ci sono
    X_cols = [c for c in df_real.columns if c != f"{outcome_name}_real"]
    if X_cols:
        df_merged = pd.merge(df_real, df_sim, on=X_cols, how='inner')
        if df_merged.empty:
            print(f"WARNING: Merge vuoto per {outcome_name} nel file {sim_file}")
            return np.nan, np.nan, np.nan
    else:
        df_merged = pd.concat([df_real[f"{outcome_name}_real"].reset_index(drop=True),
                               df_sim[f"{outcome_name}_sim"].reset_index(drop=True)], axis=1)
    
    # MAE
    mae = mean_absolute_error(df_merged[f"{outcome_name}_real"], df_merged[f"{outcome_name}_sim"])
    
    # KS test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ks_stat, ks_pval = ks_2samp(df_merged[f"{outcome_name}_real"], df_merged[f"{outcome_name}_sim"])
    
    return mae, ks_stat, ks_pval

# ------------------------------ Script principale ------------------------------
sim_files_size = glob.glob(os.path.join(sim_folder, "simulated_r_SIZE_DAG*.csv"))
sim_files_pdi = glob.glob(os.path.join(sim_folder, "simulated_r_PDI_DAG*.csv"))

sim_size_dict = {os.path.splitext(os.path.basename(f))[0].split("DAG")[-1]: f for f in sim_files_size}
sim_pdi_dict = {os.path.splitext(os.path.basename(f))[0].split("DAG")[-1]: f for f in sim_files_pdi}

common_dags = sorted(set(sim_size_dict.keys()) & set(sim_pdi_dict.keys()), key=int)

results = []

with open(log_file, "w") as f:
    for dag_number in common_dags:
        # SIZE
        mae_size, ks_size, ks_pval_size = compute_metrics(sim_size_dict[dag_number], real_csv_size, "SIZE")
        # PDI
        mae_pdi, ks_pdi, ks_pval_pdi = compute_metrics(sim_pdi_dict[dag_number], real_csv_pdi, "PDI")
        
        # Media MAE e media KS
        mae_media = (mae_size + mae_pdi) / 2
        ks_media = (ks_size + ks_pdi) / 2
        
        results.append((dag_number, mae_size, mae_pdi, mae_media, ks_size, ks_pdi, ks_media))
        
        f.write(f"DAG {dag_number}:\n")
        f.write(f"  MAE: SIZE={mae_size:.4f}, PDI={mae_pdi:.4f}, media={mae_media:.4f}\n")
        f.write(f"  KS: SIZE={ks_size:.4f} (p={ks_pval_size:.4f}), PDI={ks_pdi:.4f} (p={ks_pval_pdi:.4f}), media={ks_media:.4f}\n\n")
    
    # Top 3 per MAE medio
    results_sorted_mae = sorted(results, key=lambda x: x[3])
    f.write("=== TOP 3 DAG con MAE medio più basso ===\n\n")
    for rank, res in enumerate(results_sorted_mae[:3], start=1):
        f.write(f"{rank} - DAG {res[0]}: MAE medio = {res[3]:.4f} (SIZE={res[1]:.4f}, PDI={res[2]:.4f})\n")
    
    f.write("\n")
    
    # Top 3 per KS medio (minore distanza KS)
    results_sorted_ks = sorted(results, key=lambda x: x[6])
    f.write("=== TOP 3 DAG con KS medio più basso ===\n\n")
    for rank, res in enumerate(results_sorted_ks[:3], start=1):
        f.write(f"{rank} - DAG {res[0]}: KS medio = {res[6]:.4f} (SIZE={res[4]:.4f}, PDI={res[5]:.4f})\n")

