import re
import sys
import joblib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import shap
import seaborn as sns
import re
import pandas as pd



################### SCRIPT PER LE VARIABILI: CHIP, AQUEOUS, ML (categoriche) #########################



def read_shap_from_log(target, log_file="_Logs/analisi_shap.log"):
    with open(log_file, "r") as f:
        lines = f.readlines()
    
    start = None
    end = None
    
    for i, line in enumerate(lines):
        if f"Importanza media per {target}:" in line:
            start = i + 1
        if target == "SIZE" and "Importanza media per PDI:" in line:
            end = i
            break
    
    if start is None:
        return pd.DataFrame(columns=["Feature", "Importance"])  
    
    if end is None:
        end = len(lines)
    
    table_lines = [line.strip() for line in lines[start:end] if line.strip() != ""]
    
    data = []
    for row in table_lines:
        if re.match(r"^\d+\s+\w+", row):
            parts = re.split(r"\s+", row)
            data.append([parts[1], float(parts[2])])
    
    return pd.DataFrame(data, columns=["Feature", "Importance"])


def calcola_prob_variabile(dag_edges, df, target_var):

    dag = nx.DiGraph(dag_edges)

    parents = list(dag.predecessors(target_var))

    group_cols = parents + [target_var] if parents else [target_var]
    group = df.groupby(group_cols).size().reset_index(name='conteggio')

    if parents:
        totali = df.groupby(parents).size().reset_index(name='totale')
        group = pd.merge(group, totali, on=parents)
        group[f"P_{target_var}"] = group['conteggio'] / group['totale']
    else:
        group[f"P_{target_var}"] = group['conteggio'] / len(df)

   
    group = group.drop(columns=['conteggio', 'totale'], errors='ignore')
    return group


def propagate_intervention(df, dag_edges, intervention_var, exclude):

    #Propaga l'effetto dell'intervento lungo i figli del DAG, escludendo le variabili target.
    
    df_new = df.copy()
    dag = nx.DiGraph(dag_edges)
    to_update = [v for v in nx.descendants(dag, intervention_var) if v not in exclude]

    # Ordine topologico per garantire propagazione corretta
    topo_order = list(nx.topological_sort(dag))

    for node in topo_order:
        if node not in to_update:
            continue

        # Calcola probabilità condizionata P(node | genitori)
        parents = list(dag.predecessors(node))
        if parents:
            group = df_new.groupby(parents + [node]).size().reset_index(name='count')
            totals = df_new.groupby(parents).size().reset_index(name='total')
            group = pd.merge(group, totals, on=parents)
            group[f"P_{node}"] = group['count'] / group['total']
        else:
            counts = df_new[node].value_counts(normalize=True).reset_index()
            counts.columns = [node, f"P_{node}"]
            group = counts

        
        sampled_values = []
        for idx, row in df_new.iterrows():
            cond = group
            for p in parents:
                cond = cond[cond[p] == row[p]]

            values = cond[node].values
            probs = cond[f"P_{node}"].values
            probs = probs / probs.sum()

            # Campiona
            sampled_val = np.random.choice(values, p=probs)
            sampled_values.append(sampled_val)

        df_new[node] = sampled_values

    return df_new


df_ML = pd.read_csv("Data_Droplet/seed_non_ordinato.csv").dropna()
df = pd.read_csv("2_DAG/seed_Binn.csv").dropna()
dag_edges = [('ESM','ML'), ('HSPC','ML'), ('PEG','HSPC'), ('CHOL','PDI'), ('ML','PDI'), ('ML','SIZE'), ('CHIP','SIZE'), ('CHIP','PDI'), ('TFR','CHIP'), ('FRR','TFR'), ('FRR','CHIP'), ('AQUEOUS','CHIP'), ('CHOL','PEG'), ('AQUEOUS','TFR')]
intervention_variable= "CHIP"
nodes = ["PDI", "SIZE"]
sys.stdout = open(f"6_ANALISI_INTERVENTISTA/_log/intervention_{intervention_variable}.log", "w")

for n in nodes:
    target_var=n
    prob_fatt = calcola_prob_variabile(dag_edges, df, target_var)
    filename = f"6_ANALISI_INTERVENTISTA/_csv/Legge_congiunta_originaria_{n}.csv"
    prob_fatt.to_csv(filename, index=False)

modello_file = "6_ANALISI_INTERVENTISTA/model_rf.pkl"
modello_rf = joblib.load(modello_file)

dag = nx.DiGraph(dag_edges)
parents_esm = list(dag.predecessors(intervention_variable))

dag_do_esm = dag.copy()
dag_do_esm.remove_edges_from([(p, intervention_variable) for p in parents_esm])


ml_num_to_cat = {0: "Micromixer", 1: "Droplet"}
ml_cat_to_num = {v: k for k, v in ml_num_to_cat.items()}

x_intervento=[0.0, 1.0]
i=0
for x in x_intervento:
    i+=1
    df_intervento = df.copy()
    df_intervento[intervention_variable] = x
    df_intervento_finale = propagate_intervention(df_intervento, dag_do_esm, intervention_variable, exclude=["SIZE", "PDI"])


    df_intervento_ML = df_ML.copy()
    df_intervento_ML[intervention_variable] = ml_num_to_cat[x]

    df_intervento_ML_final = propagate_intervention(df_intervento_ML, dag_do_esm, intervention_variable, exclude=["SIZE", "PDI"])
    
    #Nuova predizione
    y_pred_intervento = modello_rf.predict(df_intervento_ML)

    df_expected_ml = df_intervento_ML_final.copy()
    df_expected_ml["PDI_pred"] = y_pred_intervento[:, 1] if y_pred_intervento.ndim > 1 else y_pred_intervento
    df_expected_ml["SIZE_pred"] = y_pred_intervento[:, 0] if y_pred_intervento.ndim > 1 else y_pred_intervento
    filename_expected_ml = f"6_ANALISI_INTERVENTISTA/_csv/Expected_ML_I{i}_{intervention_variable}.csv"
    df_expected_ml.to_csv(filename_expected_ml, index=False)

    for n in nodes:
        print(f"\n{n}, do ({intervention_variable}= {x})")

        expected_dag = calcola_prob_variabile(dag_do_esm, df_intervento_finale, n)

        filename_exp = f"6_ANALISI_INTERVENTISTA/_csv/Expected_DAG_I{i}_{intervention_variable}_{n}.csv"
        expected_dag.to_csv(filename_exp, index=False)

        print(f"INTERVENTO {i}")


        # DIREZIONE DELL'EFFETTO E PESATURA con ERRORI (DIFFERENZE)
        media_do_x = (expected_dag[n] * expected_dag[f"P_{n}"]).sum()
        prob_fatt = calcola_prob_variabile(dag_edges, df, n)
        media_target_ml = pd.read_csv(f"6_ANALISI_INTERVENTISTA/_csv/Expected_ML_I{i}_{intervention_variable}.csv")[f"{n}_pred"].mean()

        media_baseline_dag = (prob_fatt[n] * prob_fatt[f"P_{n}"]).sum()
        media_baseline_ml = pd.read_csv("6_ANALISI_INTERVENTISTA/_csv/Predizione_ML_originaria.csv")[f"{n}_pred"].mean()
        
        
        print(f"media predizione {n} baseline ML:", media_baseline_ml, f"media predizione {n} ML:", media_target_ml)
        print(f"media {n} baseline dag:",media_baseline_dag, f"media {n} dag do(X):", media_do_x)

        
        delta_DAG = media_do_x - media_baseline_dag #ACE
        delta_ML = media_target_ml - media_baseline_ml

        print(f"delta pre-post (dag)", delta_DAG, "delta pre-post (ML)", delta_ML)
        
        # MAGNITUDINE
        baseline_diff = media_baseline_ml - media_baseline_dag
        post_diff = media_target_ml - media_do_x
        delta_diff = post_diff - baseline_diff

        print("errore medie pre intervento:", baseline_diff, "errore medie post intervento:", post_diff)


        percentuale_delta_DAG= (delta_DAG/media_baseline_dag)*100
        percentuale_delta_ML= (delta_ML/media_baseline_ml)*100
        diff_perc=abs(percentuale_delta_DAG-percentuale_delta_ML)

        tolleranza=0.05
        print("percentuale delta_dag", percentuale_delta_DAG,"percentuale delta_ml", percentuale_delta_ML)
        print("differenza tra le percentuali dei delta:", diff_perc)

        print("criterio decisionale magnitudo:", diff_perc / 100, "<", tolleranza + abs(delta_diff / baseline_diff))

        #ANALISI
        if delta_DAG * delta_ML >= 0 and diff_perc / 100 < tolleranza + abs(delta_diff / baseline_diff):
            if delta_DAG*delta_ML==0:
                print(f" INTERVENTO {i}: Almeno uno dei due è zero")
        
            print (f" INTERVENTO {i}: Il modello predittivo risponde bene all'intervento causale")
        else: 
            print(f" INTERVENTO {i}: Il modello predittivo NON risponde bene all'intervento causale --> Eseguire analisi SHAP")
            
            ######## ANALISI SHAP ###########
            X = df_intervento_ML.drop(columns=n)

            if 'preprocessor' in modello_rf.named_steps:
                preprocessor = modello_rf.named_steps['preprocessor']
                X_transformed = preprocessor.transform(X)
                feature_names_transformed = preprocessor.get_feature_names_out()
            else:
                X_transformed = X.values
                feature_names_transformed = X.columns

            # Seleziona il modello corretto
            if n == "SIZE":
                rf_model = modello_rf.named_steps['regressor'].estimators_[0]
            elif n == "PDI":
                rf_model = modello_rf.named_steps['regressor'].estimators_[1]
            else:
                raise ValueError(f"Target {n} non riconosciuto")

            # Feature map
            feature_map = {
                "cat__ML_ESM": "ML",
                "cat__ML_HSPC": "ML",
                "cat__CHIP_Droplet": "CHIP",
                "cat__CHIP_Micromixer": "CHIP",
                "cat__OUTPUT_YES": "OUTPUT",
                "cat__OUTPUT_NO": "OUTPUT",
                "cat__AQUEOUS_MQ": "AQUEOUS",
                "cat__AQUEOUS_PBS": "AQUEOUS",
                "remainder__ESM": "ESM",
                "remainder__CHOL": "CHOL",
                "remainder__FRR": "FRR",
                "remainder__TFR": "TFR",
                "remainder__HSPC": "HSPC",
                "remainder__PEG": "PEG",
            }

            unique_features = list(set(feature_map.values()))

            # Calcola SHAP
            explainer = shap.TreeExplainer(rf_model)
            shap_values = explainer.shap_values(X_transformed)

            def aggregate_shap(shap_vals, feature_names_transformed, feature_map):
                unique_features = list(set(feature_map.values()))
                shap_agg = np.zeros((shap_vals.shape[0], len(unique_features)))
                for i, col in enumerate(feature_names_transformed):
                    orig_feat = feature_map.get(col, col)
                    idx = unique_features.index(orig_feat)
                    shap_agg[:, idx] += shap_vals[:, i]
                return shap_agg, unique_features

            shap_values_agg, agg_features = aggregate_shap(shap_values, feature_names_transformed, feature_map)

            shap_importance = np.abs(shap_values_agg).mean(axis=0)
            shap_importance_df = pd.DataFrame({
                'Feature': agg_features,
                'Importance': shap_importance
            }).sort_values(by='Importance', ascending=False)


            shap_original_df = pd.DataFrame(columns=['Feature', 'Importance']) 
            shap_diff_df = pd.DataFrame(columns=['Feature', 'Importance_intervento', 'Importance_original', 'Diff', 'valutazione'])  # vuoto

            if n == "PDI":
                shap_original_df = read_shap_from_log("PDI")
            elif n == "SIZE":
                shap_original_df = read_shap_from_log("SIZE")
            else:
                raise ValueError(f"Target {n} non riconosciuto")

            shap_original_df = shap_original_df[shap_original_df['Feature'].isin(shap_importance_df['Feature'])].copy()
            shap_diff_df = shap_importance_df.merge(shap_original_df, on="Feature", suffixes=("_intervento", "_original"))
            shap_diff_df["Diff"] = shap_diff_df["Importance_intervento"] - shap_diff_df["Importance_original"]

            shap_diff_df["valutazione"] = shap_diff_df.apply(
                lambda row: "correlazione spuria" if row["Diff"] / row["Importance_original"] >= 0.05 else "",
                axis=1
            )

            print(f"Differenza valori SHAP per {n} con valutazione:")
            print(shap_diff_df.sort_values("Diff", ascending=False))


    # ===== HEATMAP DELLE CORRELAZIONI =====
    # Heatmap sul dataset originale
    plt.figure(figsize=(10, 8))
    corr_matrix_original = df.corr()
    sns.heatmap(
        corr_matrix_original,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar=True,
        square=True
    )
    plt.title("Correlation Heatmap - Dataset Originale")
    plt.tight_layout()
    plt.savefig("6_ANALISI_INTERVENTISTA/_plot/correlation_heatmap_original.png")
    
    # Heatmap sul dataset post-intervento 
    plt.figure(figsize=(10, 8))
    corr_matrix_intervento = df_intervento_finale.select_dtypes(include=[np.number]).corr()
    sns.heatmap(
        corr_matrix_intervento,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar=True,
        square=True
    )
    plt.title("Correlation Heatmap - Dataset Post-Intervento")
    plt.tight_layout()
    plt.savefig(f"6_ANALISI_INTERVENTISTA/_plot/correlation_heatmap_intervento{i}_{intervention_variable}.png")



       