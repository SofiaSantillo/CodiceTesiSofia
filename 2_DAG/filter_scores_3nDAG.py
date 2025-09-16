import pandas as pd
import re

log_file_path = "2_DAG/_Logs/3nDAG_valid_with_scores.log"

dag_data = []

with open(log_file_path, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            dag_id, rest = line.split(":", 1)
            edges_part, scores_part = rest.split("|", 1)
            edges = edges_part.strip()
            
            similarity = float(re.search(r"similarity_score:\s*([0-9.]+)", scores_part).group(1))
            ll = float(re.search(r"LL:\s*([-0-9.]+)", scores_part).group(1))
            aic = float(re.search(r"AIC:\s*([0-9.]+)", scores_part).group(1))
            bic = float(re.search(r"BIC:\s*([0-9.]+)", scores_part).group(1))

            dag_data.append({
                "dag_id": dag_id.strip(),
                "edges": edges,
                "BIC": bic,
                "AIC": aic,
                "LL": ll,
                "similarity": similarity
            })
        except Exception as e:
            print(f"Errore parsing linea: {line}\n{e}")

df_dags = pd.DataFrame(dag_data)

# --- Ordina secondo la gerarchia: ---
# BIC (min), AIC (min), LL (max), similarity (max)
df_sorted = df_dags.sort_values(
    by=["BIC", "AIC", "LL", "similarity"], 
    ascending=[True, True, False, False]
)

top_10 = df_sorted.head(10)

output_log_path = "2_DAG/_Logs/top_3nDAG.log"
with open(output_log_path, "w") as f:
    for _, row in top_10.iterrows():
        f.write(f"{row['dag_id']}: {row['edges']} | similarity_score: {row['similarity']:.4f} | LL: {row['LL']:.2f} | AIC: {row['AIC']:.2f} | BIC: {row['BIC']:.2f}\n")

print(f"Top 10 DAG salvati in {output_log_path}")
print(top_10[["dag_id", "edges", "BIC", "AIC", "LL", "similarity"]])
