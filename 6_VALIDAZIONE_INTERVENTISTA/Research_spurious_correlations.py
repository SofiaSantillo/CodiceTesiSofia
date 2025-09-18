import os
import glob
import subprocess
import pandas as pd

# --- Flag per eseguire gli script ---
RUN_FEATURE = False
RUN_ACE = False
RUN_VALIDATION = False
RUN_SHAP = False


# --- Lista degli script e flag ---
SCRIPTS = [
    ("6_VALIDAZIONE_INTERVENTISTA/Feature_structure_DAG.py", RUN_FEATURE),
    ("6_VALIDAZIONE_INTERVENTISTA/Continuous_TCE.py", RUN_ACE),
    ("6_VALIDAZIONE_INTERVENTISTA/Sensitivity_Shuffle.py", RUN_VALIDATION),
    ("6_VALIDAZIONE_INTERVENTISTA/Shap_analysis.py", RUN_SHAP),
]

# -------------------------------
for script_path, run_flag in SCRIPTS:
    if run_flag:
        print(f"Esecuzione: {script_path}")
        try:
            # subprocess.run richiede il percorso corretto, Python 3.10+ consente check=True
            result = subprocess.run(
                ["python", script_path],
                capture_output=True,
                text=True,
                check=True
            )
            print(result.stdout)
            if result.stderr:
                print("Errori:", result.stderr)
        except subprocess.CalledProcessError as e:
            print(f"Errore durante l'esecuzione di {script_path}")
            print(e.stderr)
    else:
        print(f"Saltato (flag=False): {script_path}")

print("Tutti gli script eseguiti secondo i flag.")



# -------------------------------
# 1. Leggi il log originale
# -------------------------------
log_file = "6_VALIDAZIONE_INTERVENTISTA/_log/Complete_Analysis.log"

# Leggi il file con header
df = pd.read_csv(log_file, sep="|", header=0, engine="python")

# Rimuovi spazi dai nomi delle colonne
df.columns = df.columns.str.strip()

# Rimuovi spazi anche nei valori stringa
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Forza numeriche le colonne attese
for col in ["TCE_Value", "Percentuale", "Sensitivity", "ShuffleDelta", "SHAP"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# -------------------------------
# 2. Calcolo percentili
# -------------------------------
df["tce_abs"] = df["TCE_Value"].abs()
df["tce_abs_pct"] = df.groupby("Target")["tce_abs"].rank(pct=True)
df["sens_pct"] = df.groupby("Target")["Sensitivity"].rank(pct=True)
df["shuffle_pct"] = df.groupby("Target")["ShuffleDelta"].rank(pct=True)
df["shap_pct"] = df.groupby("Target")["SHAP"].rank(pct=True)

# -------------------------------
# 3. Regola di coerenza spurious_flag
# -------------------------------

tce_threshold = 0.1  # ad esempio, TCE assoluto <10% è trascurabile
sens_threshold = 0.3  # percentili per sensibilità/SHAP/Shuffle


def compute_spurious(row):
    if row["DAG_Category"] in ["direct", "indirect"]:
        # TCE alto e metriche basse -> possibile spurio
        if abs(row["TCE_Value"]) > tce_threshold and (
            row["sens_pct"] < sens_threshold or
            row["shuffle_pct"] < sens_threshold or
            row["shap_pct"] < sens_threshold
        ):
            return True
        # TCE basso ma metriche alte -> spurio
        if abs(row["TCE_Value"]) <= tce_threshold and (
            row["sens_pct"] > 1 - sens_threshold or
            row["shuffle_pct"] > 1 - sens_threshold or
            row["shap_pct"] > 1 - sens_threshold
        ):
            return True
        return False

    elif row["DAG_Category"] == "independent":
        # Feature indipendente: se metriche tutte alte -> spurio
        if row["sens_pct"] > 1 - sens_threshold or \
           row["shuffle_pct"] > 1 - sens_threshold or \
           row["shap_pct"] > 1 - sens_threshold:
            return True
        return False

    elif row["DAG_Category"] == "target":
        return False

    return False


df["spurious_flag"] = df.apply(compute_spurious, axis=1)

# -------------------------------
# 4. Scrittura su file log stile Complete_Analysis
# -------------------------------
output_file = "6_VALIDAZIONE_INTERVENTISTA/_log/MASTER_LOG_xgb_pdi.log"

with open(output_file, "w") as f:
    # intestazione
    f.write(
        f"{'Target':<12}| {'Feature':<17}| {'DAG_Category':<13}| {'TCE_Value':<12}| "
        f"{'Percentuale':<12}| {'Sensitivity':<11}| {'ShuffleDelta':<12}| {'SHAP':<11}| {'Spurious_Flag':<14}\n"
    )

    # righe
    for _, row in df.iterrows():
        tce_val = "" if pd.isna(row["TCE_Value"]) else f"{row['TCE_Value']:.6f}"
        pct_val = "" if pd.isna(row["Percentuale"]) else f"{row['Percentuale']:.6f}"
        sens_val = "" if pd.isna(row["Sensitivity"]) else f"{row['Sensitivity']:.6f}"
        shuf_val = "" if pd.isna(row["ShuffleDelta"]) else f"{row['ShuffleDelta']:.6f}"
        shap_val = "" if pd.isna(row["SHAP"]) else f"{row['SHAP']:.6f}"

        f.write(
            f"{row['Target']:<12}| {row['Feature']:<17}| {row['DAG_Category']:<13}| "
            f"{tce_val:<12}| {pct_val:<12}| {sens_val:<11}| "
            f"{shuf_val:<12}| {shap_val:<11}| {str(row['spurious_flag']):<13}\n"
        )

print(f"Log scritto in {output_file}")

log_file = "6_VALIDAZIONE_INTERVENTISTA/_log/MASTER_LOG_xgb_pdi.log"

# Leggi tutto il file
with open(log_file, "r") as f:
    lines = f.readlines()

# Sostituisci la seconda riga con trattini
lines[1] = "-" * 130 + "\n"  # 100 trattini, regola la lunghezza se vuoi

# Riscrivi il file aggiornato
with open(log_file, "w") as f:
    f.writelines(lines)
