import pandas as pd

def bin_SIZE(df, logger):
    bins_SIZE = [0, 100, 150, 200, df['SIZE'].max() + 1]
    df['SIZE_CAT'] = pd.cut(df['SIZE'], bins=bins_SIZE, labels=False, include_lowest=True)

    logger.info(f"Numero di bin per 'SIZE': {len(bins_SIZE) - 1}")
    logger.info("Conteggio degli elementi in ciascun bin di 'SIZE_CAT':")
    logger.info(f"{df['SIZE_CAT'].value_counts(sort=False)}")
    for i in range(len(bins_SIZE) - 1):
        logger.info(f"Bin {i}: da {bins_SIZE[i]:.4f} a {bins_SIZE[i+1]:.4f}")

    return df