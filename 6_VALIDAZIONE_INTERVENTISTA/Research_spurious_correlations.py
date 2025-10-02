import pandas as pd

dag_csv = "6_VALIDAZIONE_INTERVENTISTA/_csv/analisi_causale.csv"
feat_csv = "6_VALIDAZIONE_INTERVENTISTA/_csv/analisi_correlativa.csv"
output_csv = "6_VALIDAZIONE_INTERVENTISTA/_csv/Spurious_Correletions.csv"
dag_df = pd.read_csv(dag_csv)
feat_df = pd.read_csv(feat_csv)
dag_df.columns = dag_df.columns.str.strip()
feat_df.columns = feat_df.columns.str.strip()

merged_df = pd.merge(dag_df, feat_df, on=["Target", "Feature"], how="outer")
merged_df.to_csv(output_csv, index=False)

print(f"Merge completato. File salvato in '{output_csv}'")

import pandas as pd

merged_csv = "6_VALIDAZIONE_INTERVENTISTA/_csv/Spurious_Correletions.csv"
df = pd.read_csv(merged_csv)
df.columns = df.columns.str.strip()

rows = []

for _, row in df.iterrows():
    X = row["Feature"]
    Y = row["Target"]
    Z = row["Covariates"]
    Category=row["DAG_Category"]
    
    eff = row.get("Est_Effect", row.get("effect_adj (DAG)", float("nan")))
    sens = row.get("Sensitivity (ML)", row.get("sensitivity (ML)", float("nan")))
    sdel = row.get("Permutation Importance (ML)", row.get("Permutation Importance (ML)", float("nan")))

    
    rows.append({
        "target": Y,
        "feature": X,
        "adj_set": Z,
        "category":Category,
        "effect_adj (DAG)": eff,
        "sensitivity (ML)": sens,
        "Permutation Importance (ML)": sdel,
    })

res = pd.DataFrame(rows)
if res.empty:
    print("Nessun dato disponibile.")
else:
    res["eff_abs"] = res["effect_adj (DAG)"].abs()
    res["sens"] = res["sensitivity (ML)"].abs()
    res["perm"] = res["Permutation Importance (ML)"].abs()

    res["eff_abs_pct"] = res.groupby("target")["eff_abs"].rank(pct=True)
    res["sens_pct"] = res.groupby("target")["sens"].rank(pct=True)
    res["PI_pct"]   = res.groupby("target")["perm"].rank(pct=True)

    res["spurious_flag"] = (
        (res["eff_abs_pct"] <= 0.40) &
        ((res["sens_pct"] >= 0.60) | (res["PI_pct"] >= 0.60))
    )
    res["segno"] = (res["effect_adj (DAG)"] * res["sensitivity (ML)"] > 0)

res.to_csv("6_VALIDAZIONE_INTERVENTISTA/_csv/Spurious_Correletions.csv", index=False)

log_file = "6_VALIDAZIONE_INTERVENTISTA/_log/Spurious_Correletions.log"
csv_file = "6_VALIDAZIONE_INTERVENTISTA/_csv/Spurious_Correletions.csv"

df = pd.read_csv(csv_file)
df.columns = df.columns.str.strip()

# -------------------------------
# Scrittura log in formato tabellare
# -------------------------------
with open(log_file, "w") as f:
    f.write(
        f"{'Target':<10} | {'Feature':<10} | {'DAG_Category':<13} | {'Covariates':<10} | {'Adj_Effect (DAG)':<22} | {'Sensitivity (ML)':<18} | {'Permutation Importance (ML)':<28} | {'Spurious_Flag':<12}| {'Coerenza causale':<12}\n"
    )
    f.write("-" * 160 + "\n")

    for _, row in df.iterrows():
        f.write(
            f"{str(row.get('target','')):<10} | "
            f"{str(row.get('feature','')):<10} | "
            f"{str(row.get('category','')):<13} | "
            f"{str(row.get('adj_set','')):<10} | "
            f"{str(row.get('effect_adj (DAG)','')):<22} | "
            f"{str(row.get('sensitivity (ML)','')):<18} | "
            f"{str(row.get('Permutation Importance (ML)','')):<28} | "
            f"{str(row.get('spurious_flag','')):<12} | "
            f"{str(row.get('segno','')):<12}\n"
        )

print(f"File log salvato in '{log_file}'")