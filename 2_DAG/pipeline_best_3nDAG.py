import subprocess

# Flag: True per eseguire, False per saltare
RUN_SCRIPT_1 = False
RUN_SCRIPT_2 = False
RUN_SCRIPT_3 = False
RUN_SCRIPT_4 = True
RUN_SCRIPT_5 = True

SCRIPTS = [
    ("2_DAG/generate_all_combination_of_3_nodes.py", RUN_SCRIPT_1),
    ("2_DAG/creation_3nDAG.py", RUN_SCRIPT_2),
    ("2_DAG/calculate_scores_3nDAG.py", RUN_SCRIPT_3),
    ("2_DAG/filter_vincoli_3nDAG.py", RUN_SCRIPT_4),
    ("2_DAG/filter_scores_3nDAG.py", RUN_SCRIPT_5),
]

for script, flag in SCRIPTS:
    if flag:
        print(f"🔹 Eseguo {script}...")
        result = subprocess.run(["python", script], capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errore:", result.stderr)
    else:
        print(f"Salto {script}")
