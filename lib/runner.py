import pandas as pd
import numpy as np
import time
from sklearn.metrics import f1_score, recall_score, roc_curve, auc, accuracy_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold 
from tqdm.auto import tqdm
from memory_profiler import memory_usage

from lib.util import (
    RANDOM_STATE, RESULTS_PATH, RESULTS_DIR, 
    load_json, save_json, save_model, load_model, save_results
)

N_SPLITS = 5

def fit_model(model, X, y):
    initial_memory = [memory_usage(-1, interval=0.1, max_usage=True)]
    initial_time = time.time()
    mem_usage, _ = memory_usage((model.fit, (X, y)), retval=True, interval=0.1)
    final_time = time.time() - initial_time
    mem_usage = [x - initial_memory[0] for x in mem_usage]  # Ajusta consumo de memória real
    return max(mem_usage), sum(mem_usage) / max(len(mem_usage), 1), final_time

def evaluate_model(model, X_train, Y_train, X_test, Y_test, k) -> pd.DataFrame:
    metrics = {"K": [k]}
    for dataset, X, Y in [("train", X_train, Y_train), ("test", X_test, Y_test)]:
        Y_pred = model.predict(X)
        metrics[f"F1_Score_{dataset}"] = [f1_score(Y, Y_pred)]
        metrics[f"Recall_{dataset}"] = [recall_score(Y, Y_pred)]
        metrics[f"AUC_{dataset}"] = [auc(*roc_curve(Y, Y_pred)[:2])]
        metrics[f"Accuracy_{dataset}"] = [accuracy_score(Y, Y_pred)]
        if dataset == "test":
            fpr, tpr, _ = roc_curve(Y, Y_pred)
            metrics[f"FPR_{dataset}"] = [list(fpr)]
            metrics[f"TPR_{dataset}"] = [list(tpr)]
    return pd.DataFrame(metrics)

def new_search_params(model, params: dict, X_train, Y_train, model_name: str, max_combinations=np.inf, stop_iter: int = None, load_model: bool = True, save: bool = True):
    models_results = load_json(RESULTS_PATH)
    if load_model and model_name in models_results:
        print(f"Modelo {model_name} encontrado. Carregando...")
        best_params = models_results[model_name]['best_params']
        best_model = load_model(model_name)
        
        if best_model is None:
            print(f"Modelo {model_name} não encontrado no disco. Treinando um novo modelo...")
            best_model = model.set_params(**best_params)
            best_model.fit(X_train, Y_train)
            save_model(best_model, model_name)
        
        df_final = pd.read_csv(models_results[model_name]['result'])
        df_iter = None
        results = models_results[model_name]
        return df_final, best_model, results, df_iter
    
    if (num_combinations := min(np.prod([len(v) for v in params.values()]), max_combinations)) < 20:
        raise ValueError(f"O número de combinações ({num_combinations}) é menor que 20. Ajuste os hiperparâmetros.")
    
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    
    print("Iniciando busca por hiperparâmetros...")
    search_time_start = time.time()
    search_model = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        cv=3,
        n_iter=num_combinations,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    
    max_mem, avg_mem, fit_time = fit_model(search_model, X_train, Y_train)
    search_time_end = time.time() - search_time_start
    
    best_params = search_model.best_params_
    best_model = model.set_params(**best_params)
    
    kfold = StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_STATE, shuffle=True)
    results_list = []
    iter_list = []
    
    for k, (train_idx, test_idx) in tqdm(enumerate(kfold.split(X_train, Y_train), start=1), total=N_SPLITS, desc=f"Cross-Validation ({N_SPLITS}-folds)"):
        X_train_fold, X_valid_fold = X_train[train_idx], X_train[test_idx]
        Y_train_fold, Y_valid_fold = Y_train[train_idx], Y_train[test_idx]
        
        best_model.fit(X_train_fold, Y_train_fold)
        df_results = evaluate_model(best_model, X_train_fold, Y_train_fold, X_valid_fold, Y_valid_fold, k)
        results_list.append(df_results)
        
        if stop_iter:
            iter_list.append(df_results.mean(numeric_only=True))
            if len(iter_list) > stop_iter and iter_list[-1]['F1_Score_test'] < iter_list[-stop_iter]['F1_Score_test']:
                print(f"Parando treinamento após {stop_iter} iterações devido à deterioração do desempenho.")
                break
    
    df_final = pd.concat(results_list, ignore_index=True)
    df_final = df_final.sort_values(by="F1_Score_test", ascending=False)
    df_final = df_final[["K", "AUC_test", "AUC_train", "Accuracy_test", "Accuracy_train", "F1_Score_test", "F1_Score_train", "Recall_test", "Recall_train", "FPR_test", "TPR_test"]]
    df_iter = pd.DataFrame(iter_list) if stop_iter else None
    
    if save:
        model_path = save_model(best_model, model_name)
        results_path = save_results(df_final, model_name)
        
        results = {
            "model_name" : model_name,
            "search_execution_time": search_time_end,
            "fit_execution_time": fit_time,
            "memory_max": max_mem,
            "memory_avg": avg_mem,
            "best_params": best_params,
            "model": model_path,
            "result": results_path
        }
        models_results[model_name] = results
        save_json(models_results, RESULTS_PATH)
    
    return df_final, best_model, results, df_iter
