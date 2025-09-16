import subprocess
import os
import re
import pandas as pd
from collections import Counter

# ----------------------- Setup -----------------------
log_folder = "4_SAMPLING/_logs"
best3_file = os.path.join(log_folder, "comparison_sampling.log")  # log generato da compare_r.py
run_log_file = os.path.join(log_folder, "run_iterations.log")

os.makedirs(log_folder, exist_ok=True)

# ----------------------- Flag script -----------------------
RUN_ANCESTRAL = True
RUN_SIMULATED = True
RUN_REAL = True
RUN_COMPARE = True

SCRIPTS = [
    ("4_SAMPLING/ancestral_sampling.py", RUN_ANCESTRAL),
    ("4_SAMPLING/simulated_r.py", RUN_SIMULATED),
    ("4_SAMPLING/real_r.py", RUN_REAL),
    ("4_SAMPLING/compare_r.py", RUN_COMPARE),
]

# ----------------------- Inizio nuovo run -----------------------
with open(run_log_file, "w") as log:
    log.write("=== Avvio nuovo run completo ===\n\n")

all_top3_dags = []  # Lista per salvare i top3 di ogni iterazione

# ----------------------- Esecuzione iterazioni -----------------------

for script_path, flag in SCRIPTS:
    with open(run_log_file, "a") as log:
        if flag:
            log.write(f"Eseguo {script_path}...\n")
            print(f"Eseguo {script_path}...")
            try:
                subprocess.run(["python", script_path], check=True)
                log.write(f"{script_path} eseguito con SUCCESSO.\n")
                print(f"{script_path} SUCCESS\n")
            except subprocess.CalledProcessError:
                log.write(f"{script_path} ha FALLITO.\n")
                print(f"{script_path} FALLITO\n")
        else:
            log.write(f"Salto {script_path} (flag=False).\n")
            print(f"Salto {script_path} (flag=False).")

    # ----------------------- Estrazione DAG con MAE e KS -----------------------
dag_metrics = []
if os.path.exists(best3_file):
    with open(best3_file, "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        dag_match = re.match(r"DAG\s*(\d+):", line)
        if dag_match:
            dag_number = dag_match.group(1)
            try:
                mae_line = lines[i + 1].strip()
                ks_line = lines[i + 2].strip()
                # Estrazione MAE medio
                mae_match = re.search(r"media\s*=\s*([\d\.]+)", mae_line)
                ks_match = re.search(r"SIZE=([\d\.]+).*PDI=([\d\.]+)", ks_line)
                if mae_match and ks_match:
                    mae_value = float(mae_match.group(1))
                    ks_size = float(ks_match.group(1))
                    ks_pdi = float(ks_match.group(2))
                    ks_value = (ks_size + ks_pdi) / 2  # KS medio
                    dag_metrics.append((dag_number, mae_value, ks_value))
            except IndexError:
                continue

    # ----------------------- Calcolo tradeoff -----------------------
    if dag_metrics:
        df = pd.DataFrame(dag_metrics, columns=["DAG", "MAE", "KS"])
        # Normalizzazione MAE e KS
        df["MAE_norm"] = (df["MAE"] - df["MAE"].min()) / (df["MAE"].max() - df["MAE"].min())
        df["KS_norm"] = (df["KS"] - df["KS"].min()) / (df["KS"].max() - df["KS"].min())
        # Tradeoff
        df["Tradeoff"] = df["MAE_norm"] + df["KS_norm"]
        top3_tradeoff = df.sort_values("Tradeoff").head(3)

        # Salvataggio top3 dell'iterazione
        with open(run_log_file, "a") as log:
            log.write(f"\n--- TOP 3 DAG con miglior tradeoff ---\n")
            for _, row in top3_tradeoff.iterrows():
                log.write(f"DAG {row['DAG']}: MAE={row['MAE']:.4f}, KS={row['KS']:.4f}, Tradeoff={row['Tradeoff']:.4f}\n")
                all_top3_dags.append(row['DAG'])
    else:
        with open(run_log_file, "a") as log:
            log.write(f"File {best3_file} non trovato.\n")


# ----------------------- Salvataggio edges dei top 3 DAG finali -----------------------
best_dags_file = "5_ACE/_logs/best_dags.log"
os.makedirs(os.path.dirname(best_dags_file), exist_ok=True)

# Calcolo dei 3 DAG più frequenti tra i top3 delle iterazioni
top3_frequent = Counter(all_top3_dags).most_common(3)

# File sorgente contenente tutti i DAG con edges
source_file = "3_D-SEPARATION/_logs/expanded_Dags_clean.log"

# Lettura del file sorgente
with open(source_file, "r") as f:
    content = f.read()

# Separazione in blocchi di DAG tramite linee di separazione
dag_blocks = re.split(r'-{30,}', content)

best_dags_info = []

for block in dag_blocks:
    dag_match = re.search(r"DAG\s*(\d+):", block)
    edges_match = re.search(r"Edges:\s*(\[[^\]]*\])", block, re.DOTALL)
    if dag_match and edges_match:
        dag_number = dag_match.group(1)
        if dag_number in [dag for dag, _ in top3_frequent]:  # solo top 3 frequenti
            edges = edges_match.group(1)
            best_dags_info.append((dag_number, edges))

# Scrittura sul nuovo file log
with open(best_dags_file, "w") as f:
    f.write("=== Top 3 DAG migliori (numero + edges) ===\n\n")
    for dag_number, edges in best_dags_info:
        f.write(f"DAG {dag_number}:\nEdges: {edges}\n\n")