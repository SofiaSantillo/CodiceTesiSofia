import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def classify_simple_dependency(df, r2_threshold=0.1, importance_threshold=0.05):
    """
    Classifica ogni variabile come 'differenziale' o 'nessuna'
    in base a quanto bene può essere predetta dalle altre variabili.
    Restituisce solo le feature influenti con importanza >= importance_threshold.
    """
    results = {}
    df_diff = df.diff().dropna()
    variables = df.columns

    for var in variables:
        X_diff = df_diff.drop(columns=[var])
        y_diff = df_diff[var]

        # Modello Random Forest
        model = RandomForestRegressor(n_estimators=100, random_state=0)
        model.fit(X_diff, y_diff)
        r2 = model.score(X_diff, y_diff)

        importances = dict(
            sorted(
                zip(X_diff.columns, model.feature_importances_),
                key=lambda x: x[1],
                reverse=True
            )
        )

        filtered_importances = {k: v for k, v in importances.items() if v >= importance_threshold}

        # Classificazione
        if r2 > r2_threshold and filtered_importances:
            dep_type = "dipendenza"
        else:
            dep_type = "nessuna"
            filtered_importances = {}

        results[var] = {
            "r2_differenziale": r2,
            "dipendenza": dep_type,
            "variabili_influenti": filtered_importances
        }

    return pd.DataFrame(results).T


df = pd.read_csv("2_DAG/seed_Binn.csv")
dep_df = classify_simple_dependency(df)
print(dep_df)

dep_df_json = dep_df.to_dict(orient="index")
with open("2_DAG/_json/dependency_results.json", "w") as f:
    json.dump(dep_df_json, f, indent=4)
