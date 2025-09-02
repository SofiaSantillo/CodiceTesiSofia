import pandas as pd
import glob
import os


input_folder = "4_SAMPLING/_csv/_csv_archive"
output_folder = "4_SAMPLING/_csv/_csv_archive"
os.makedirs(output_folder, exist_ok=True)
dataset_files = glob.glob(os.path.join(input_folder, "dataset_sampling_DAG*.csv"))

for file_path in dataset_files:
    dag_number = os.path.splitext(os.path.basename(file_path))[0].split("DAG")[-1]

    df = pd.read_csv(file_path)

    outcomes = ['SIZE', 'PDI']

    for outcome in outcomes:
        X_cols = [c for c in df.columns if c != outcome]

        rX = df.groupby(X_cols)[outcome].mean().reset_index()

        output_file = os.path.join(output_folder, f"simulated_r_{outcome}_DAG{dag_number}.csv")
        rX.to_csv(output_file, index=False)