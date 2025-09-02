import pandas as pd

def riordina_colonne(csv_file, nuovo_ordine_colonne, output_file):
    """
    Riordina le colonne di un dataset (DataFrame) secondo l'ordine fornito
    e salva il risultato in un nuovo file CSV.

    Parameters:
    - csv_file: il percorso del file CSV da caricare
    - nuovo_ordine_colonne: una lista dei nomi delle colonne nel nuovo ordine
    - output_file: il percorso del file CSV di destinazione per il dataset riordinato
    """
    # Carica il dataset
    df = pd.read_csv(csv_file)

    # Controlla che tutte le colonne indicate siano presenti nel dataset
    colonne_presenti = set(df.columns)
    colonne_richieste = set(nuovo_ordine_colonne)
    
    if not colonne_richieste.issubset(colonne_presenti):
        raise ValueError(f"Alcune colonne richieste non sono presenti nel dataset: {colonne_richieste - colonne_presenti}")

    # Riordina le colonne secondo l'ordine fornito
    df_riordinato = df[nuovo_ordine_colonne]

    for col in df_riordinato.select_dtypes(include=['object']).columns:
        df_riordinato[col], _ = pd.factorize(df_riordinato[col])

    # Salva il nuovo dataset riordinato in un file CSV
    df_riordinato.to_csv(output_file, index=False)  # Salva senza l'indice

    print(f"Dataset riordinato salvato in: {output_file}")
    return df_riordinato

# Esempio di utilizzo
if __name__ == "__main__":
    csv_file = 'Data_Droplet/seed_Binning.csv'  # Percorso del tuo file CSV
    nuovo_ordine_colonne = ['ML', 'HSPC','ESM','PEG','AQUEOUS','CHIP','TFR','CHOL','FRR','PDI','SIZE']  # Ordine che vuoi ottenere
    output_file = 'Data_Droplet/seed_Binning_ordinato.csv'  # Percorso per il nuovo file CSV

    try:
        df_riordinato = riordina_colonne(csv_file, nuovo_ordine_colonne, output_file)
        print("Colonne riordinate con successo!")
        print(df_riordinato.head())  # Stampa le prime righe del DataFrame riordinato
    except ValueError as e:
        print(e)
