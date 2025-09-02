import pandas as pd
import os


file_path = "Data_Droplet/seed_Binning_ordinato.csv"
output_folder = "4_SAMPLING/_csv"
os.makedirs(output_folder, exist_ok=True)

dag_number = os.path.splitext(os.path.basename(file_path))[0].split("DAG")[-1]

df = pd.read_csv(file_path)

outcomes = ['SIZE', 'PDI']

for outcome in outcomes:
    X_cols = [c for c in df.columns if c != outcome]

 
    rX = df.groupby(X_cols)[outcome].mean().reset_index()


    output_file = os.path.join(output_folder, f"real_r_{outcome}.csv")
    rX.to_csv(output_file, index=False)
