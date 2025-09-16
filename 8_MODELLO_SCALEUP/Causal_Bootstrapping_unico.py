import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# --- Funzione per calcolo pesi CB multivariati in maniera vettoriale ---
def compute_cb_weights_multi_vectorized(df, targets, U=[], Z=[], D=[]):
    """
    df: dataframe con X_conf e target multivariato
    targets: lista di colonne target ['SIZE', 'PDI']
    U, Z, D: liste di nomi di colonne (confounders, mediatori, decision)
    """
    df_work = df.copy()

    # Codifica le variabili categoriche (ordinal)
    cat_cols = df_work.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        enc = OrdinalEncoder()
        df_work[cat_cols] = enc.fit_transform(df_work[cat_cols])

    N = len(df_work)
    weights = np.ones(N)

    # Target multivariato come tupla
    target_vals = df_work[targets].apply(lambda row: tuple(row), axis=1)

    # --- Confounders U ---
    for u in U:
        counts = df_work[u].value_counts(normalize=True)
        weights *= df_work[u].map(lambda x: counts.get(x, 1.0 / N))

    # --- Mediatori Z ---
    for z in Z:
        def get_z_prob(row):
            mask = target_vals == tuple(row[targets])
            counts = df_work[mask][z].value_counts(normalize=True)
            return counts.get(row[z], 1.0 / N)
        weights *= df_work.apply(get_z_prob, axis=1)

    # --- Decision D ---
    for d in D:
        def get_d_prob(row):
            mask = target_vals == tuple(row[targets])
            for u in U:
                mask = mask & (df_work[u] == row[u])
            counts = df_work[mask][d].value_counts(normalize=True)
            return counts.get(row[d], 1.0 / N)
        weights *= df_work.apply(get_d_prob, axis=1)

    # Normalizza
    weights /= np.sum(weights)
    return weights

# --- Resample dataset con CB weights multivariati ---
def causal_bootstrap_resample_multi(df, targets, U=[], Z=[], D=[]):
    """
    Resample dataset usando pesi CB calcolati su target multivariato
    """
    weights = compute_cb_weights_multi_vectorized(df, targets, U, Z, D)
    N = len(df)
    sampled_idx = np.random.choice(df.index, size=N, replace=True, p=weights)
    df_resampled = df.loc[sampled_idx].reset_index(drop=True)
    return df_resampled

# --- Logging semplice ---
log_file = "8_MODELLO_SCALEUP/_log/causal_bootstrap_multivariate.log"
def log_print(*args, **kwargs):
    print(*args, **kwargs)
    with open(log_file, "a") as f:
        print(*args, **kwargs, file=f)

# --- Esempio completo ---
if __name__ == "__main__":
    df_conf = pd.read_csv("_Data/validation.csv")

    # Variabili già identificate dai tuoi script
    U_all = list(set(['ESM', 'PEG']))  # unione di U_SIZE + U_PDI
    Z_all = list(set(['ESM', 'CHOL', 'PDI', 'SIZE', 'PEG', 'HSPC', 'AQUEOUS']))  # unione Z_SIZE + Z_PDI
    D_all = []  # se ci fossero variabili decision

    targets = ['SIZE', 'PDI']

    log_print("Inizio causal bootstrap su target multivariato:", targets)
    df_deconf_multi = causal_bootstrap_resample_multi(df_conf, targets, U=U_all, Z=Z_all, D=D_all)
    log_print("Dataset de-confounded generato. Prime righe:")
    log_print(df_deconf_multi.head())
    df_deconf_multi.to_csv("_Data/validation_deconf_multi.csv", index=False)
    log_print("Dataset salvato in _Data/validation_deconf_multi.csv")
