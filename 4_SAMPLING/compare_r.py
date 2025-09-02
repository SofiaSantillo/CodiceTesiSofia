import pandas as pd
from sklearn.metrics import mean_absolute_error
import glob
import os


real_csv_size = "4_SAMPLING/_csv/real_r_SIZE.csv"
real_csv_pdi = "4_SAMPLING/_csv/real_r_PDI.csv"


sim_folder = "4_SAMPLING/_csv/"
log_folder = "4_SAMPLING/_logs"
os.makedirs(log_folder, exist_ok=True)
log_file = os.path.join(log_folder, "MAE_r_comparison.log")

def compute_mae(sim_file, real_file, outcome_name):
    df_sim = pd.read_csv(sim_file)
    df_real = pd.read_csv(real_file)
    
    df_sim = df_sim.rename(columns={outcome_name: f"{outcome_name}_sim"})
    df_real = df_real.rename(columns={outcome_name: f"{outcome_name}_real"})
    
    X_cols = [c for c in df_real.columns if c != f"{outcome_name}_real"]
    df_merged = pd.merge(df_real, df_sim, on=X_cols, how='inner')
    
    mae = mean_absolute_error(df_merged[f"{outcome_name}_real"], df_merged[f"{outcome_name}_sim"])
    return mae


sim_files_size = glob.glob(os.path.join(sim_folder, "simulated_r_SIZE_DAG*.csv"))
sim_files_pdi = glob.glob(os.path.join(sim_folder, "simulated_r_PDI_DAG*.csv"))


sim_size_dict = {os.path.splitext(os.path.basename(f))[0].split("DAG")[-1]: f for f in sim_files_size}
sim_pdi_dict = {os.path.splitext(os.path.basename(f))[0].split("DAG")[-1]: f for f in sim_files_pdi}

common_dags = sorted(set(sim_size_dict.keys()) & set(sim_pdi_dict.keys()), key=int)

results = [] 

with open(log_file, "w") as f:
    for dag_number in common_dags:
        mae_size = compute_mae(sim_size_dict[dag_number], real_csv_size, "SIZE")
        mae_pdi = compute_mae(sim_pdi_dict[dag_number], real_csv_pdi, "PDI")
        media = (mae_pdi + mae_size) / 2

        results.append((dag_number, mae_size, mae_pdi, media))


        f.write(f"DAG {dag_number}: MAE per SIZE = {mae_size:.4f}, MAE per PDI = {mae_pdi:.4f} - MAE medio = {media:.4f}\n\n \n")


    # Ordino per media crescente
    results_sorted = sorted(results, key=lambda x: x[3])

  
    top3 = results_sorted[:3]

    f.write("=== TOP 3 DAG con MAE medio piu' basso ===\n\n")
    for rank, (dag_number, mae_size, mae_pdi, media) in enumerate(top3, start=1):
        f.write(f"{rank} - DAG {dag_number}: MAE medio = {media:.4f} (SIZE={mae_size:.4f}, PDI={mae_pdi:.4f})\n")


