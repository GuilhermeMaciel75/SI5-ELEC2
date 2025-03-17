import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, recall_score, roc_curve, auc, confusion_matrix, accuracy_score
from tqdm.auto import tqdm
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold 
import pickle
import time

N_SPLITS = 5
RANDOM_STATE = 51

def evaluate_model(model, X_train, Y_train, X_test, Y_test) -> pd.DataFrame:
    metrics = []

    for dataset, X, Y in [("train", X_train, Y_train), ("test", X_test, Y_test)]:
        Y_pred = model.predict(X)
        f1 = f1_score(Y, Y_pred)
        recall = recall_score(Y, Y_pred)
        acc = accuracy_score(Y, Y_pred)
        fpr, tpr, _ = roc_curve(Y, Y_pred)
        roc_auc = auc(fpr, tpr)

        metrics.append([dataset, f1, recall, roc_auc, acc, fpr, tpr])

    df_results = pd.DataFrame(metrics, columns=["Dataset", "F1_Score", "Recall", "AUC", "Accuracy", "FPR", "TPR"])
    
    return df_results

def search_params(model, params: dict, X_train, Y_train, max_combinations=np.inf):

    num_combinations = min(np.prod([len(v) for v in params.values()]), max_combinations)
    if num_combinations < 20:
        raise ValueError(f"O número de combinações ({num_combinations}) é menor que 20. Ajuste os hiperparâmetros.")
        
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    
    kfold = StratifiedKFold(n_splits=N_SPLITS, random_state=N_SPLITS, shuffle=True)

    search_model = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print(f"Num combinações de hiperparâmetros: {num_combinations}")
    print("Iniciando busca por hiperparâmetros...")
    init_time = time.time()
    search_model.fit(X_train, Y_train)
    best_params = search_model.best_params_
    print(f"Melhor conjunto de hiperparâmetros encontrado: {best_params}")
    print(f"Tempo de busca: {time.time() - init_time:.2f}s")

    results_list = []
    best_model = None
    best_f1_score = -np.inf  

    for k, (train_idx, test_idx) in tqdm(
            enumerate(kfold.split(X_train, Y_train), start=1),
            total=N_SPLITS,
            desc=f"Cross-Validation ({N_SPLITS}-folds)",
            position=0,
            leave=True,
            dynamic_ncols=True,
        ):
        
        X_train_fold, X_valid_fold = X_train[train_idx], X_train[test_idx]
        Y_train_fold, Y_valid_fold = Y_train[train_idx], Y_train[test_idx]

        best_model_fold = model.set_params(**best_params)
        best_model_fold.fit(X_train_fold, Y_train_fold)

        df_results = evaluate_model(best_model_fold, X_train_fold, Y_train_fold, X_valid_fold, Y_valid_fold)

        df_results["K"] = k

        fpr_test = df_results.loc[df_results["Dataset"] == "test", "FPR"].values[0]
        tpr_test = df_results.loc[df_results["Dataset"] == "test", "TPR"].values[0]

        df_results = df_results.pivot_table(index="K", columns="Dataset", values=["F1_Score", "Recall", "AUC", "Accuracy"]).reset_index()
        df_results.columns = ["_".join(col).strip() for col in df_results.columns]

        df_results["FPR_test"] = [fpr_test]
        df_results["TPR_test"] = [tpr_test]

        results_list.append(df_results)

        if "F1_Score_test" in df_results.columns and df_results["F1_Score_test"].values[0] > best_f1_score:
            best_f1_score = df_results["F1_Score_test"].values[0]
            best_model = best_model_fold 

    df_final = pd.concat(results_list, ignore_index=True)

    if "F1_Score_test" in df_final.columns:
        df_final.sort_values(by="F1_Score_test", ascending=False, inplace=True)

    return df_final, best_model, best_params


def save_model(model, model_name):
    model_path = f"./models/{model_name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

def save_results(df_results, model_name):
    results_path = f"./data/results/{model_name}.csv"
    df_results.to_csv(results_path, index=False)
