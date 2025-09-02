import subprocess
import os
from collections import Counter
import re

log_folder = "4_SAMPLING/_logs"
best3_file = os.path.join(log_folder, "MAE_r_comparison0.log")


run_log_file = os.path.join(log_folder, "run_iterations0.log")


with open(run_log_file, "w") as log:
    log.write("=== Avvio nuovo run completo ===\n\n")


files = [
    "4_SAMPLING/ancestral_sampling0.py",
    "4_SAMPLING/simulated_r0.py",
    "4_SAMPLING/real_r0.py",
    "4_SAMPLING/compare_r0.py"
]

n = 0
while n < 100:
    n += 1
    with open(run_log_file, "a") as log:
        log.write(f"ITERAZIONE {n}")
        print(f"ITERAZIONE {n}")
        for f in files:
            subprocess.run(["python", f], check=True)

        if os.path.exists(best3_file):
            with open(best3_file, "r") as f_in:
                lines = f_in.readlines()

            
            last_section_start = None
            for i, line in enumerate(lines):
                if line.startswith("=== TOP 3 DAG con MAE medio piu' basso ==="):
                    last_section_start = i

            if last_section_start is not None:
                log.writelines(lines[last_section_start + 1:])
                log.write("\n")
            else:
                log.write("\n Nessuna sezione TOP 3 trovata nel log.\n")
        else:
            log.write(f"\n File {best3_file} non trovato.\n")


dag_counter = Counter()


with open(run_log_file, "r+") as log:
    for line in log:
        match = re.match(r"\d+\s*-\s*DAG\s*(\d+):", line)
        if match:
            dag_number = match.group(1)
            dag_counter[dag_number] += 1


    top3_dags = dag_counter.most_common(3)

    
    log.write("\n=== I 3 DAG più frequenti nelle top 3 ===\n")
    for rank, (dag_number, count) in enumerate(top3_dags, start=1):
        log.write(f"{rank} - DAG {dag_number}: presente {count} volta/e\n")
