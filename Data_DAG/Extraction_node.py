import pandas as pd

input_file = 'Data_Droplet/seed.csv'

# Read the CSV file
df = pd.read_csv(input_file)
columns_to_extract = ['FRR', 'CHIP', 'SIZE', 'PDI', 'TFR']
df_selected = df[columns_to_extract]

# Write to a new CSV file 
output_file = 'Data_DAG/Nodi_DAG2.csv'
df_selected.to_csv(output_file, index=False)
