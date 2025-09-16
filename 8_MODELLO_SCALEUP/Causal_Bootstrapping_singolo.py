import numpy as np
import pandas as pd

# --- Funzione kernel semplice (delta function) ---
def kernel(x, x_n, bandwidth=1e-6):
    # delta-function kernel per campionamento discreto
    return 1.0 if np.allclose(x, x_n, atol=bandwidth) else 0.0

# --- Calcolo pesi causal bootstrap per una classe y=c ---
def compute_cb_weights(df, target, class_c, U, Z, D=None):
    """
    df: dataframe con X_conf e Y_conf
    target: colonna Y
    class_c: valore di Y per cui calcolare i pesi
    U, Z, D: liste di nomi di colonne (confounders, mediatori, decision)
    """
    N = len(df)
    weights = np.zeros(N)
    X_cols = [c for c in df.columns if c != target]

    for n in range(N):
        row = df.iloc[n]
        w_n = 1.0

        # Confounders U
        for u in U:
            # P(u)
            counts = df[u].value_counts(normalize=True)
            w_n *= counts.get(row[u], 1.0 / N)

        # Mediators Z
        for z in Z:
            # P(z | y = class_c)
            counts = df[df[target] == class_c][z].value_counts(normalize=True)
            w_n *= counts.get(row[z], 1.0 / N)

        # Decision D (opzionale)
        if D is not None:
            for d in D:
                # P(d | u, y=class_c)
                subset = df[(df[target] == class_c)]
                for u in U:
                    subset = subset[subset[u] == row[u]]
                counts = subset[d].value_counts(normalize=True)
                w_n *= counts.get(row[d], 1.0 / N)

        weights[n] = w_n

    # Normalizza pesi
    weights /= np.sum(weights)
    return weights

# --- Resample del dataset con CB weights ---
def causal_bootstrap_resample(df, target, U, Z, D=None):
    classes = df[target].unique()
    sampled_rows = []

    for c in classes:
        weights = compute_cb_weights(df, target, c, U, Z, D)
        n_samples = int(len(df) / len(classes))  # approx samples per class
        sampled_idx = np.random.choice(df.index, size=n_samples, replace=True, p=weights)
        sampled_rows.append(df.loc[sampled_idx])

    return pd.concat(sampled_rows).reset_index(drop=True)

# --- Esempio di utilizzo ---
df_conf = pd.read_csv("_Data/data_1.csv")

# variabili già identificate dai tuoi script
U_SIZE, Z_SIZE, D_SIZE = ['ESM'], ['ESM', 'CHOL', 'PDI'], []
U_PDI, Z_PDI, D_PDI = ['ESM', 'PEG'], ['SIZE', 'PEG', 'ESM', 'CHOL', 'HSPC', 'AQUEOUS'], []

# Dataset de-confounded
df_size_deconf = causal_bootstrap_resample(df_conf, target='SIZE', U=U_SIZE, Z=Z_SIZE, D=D_SIZE)
df_pdi_deconf  = causal_bootstrap_resample(df_conf, target='PDI',  U=U_PDI, Z=Z_PDI, D=D_PDI)

print(df_pdi_deconf)
print(df_size_deconf)