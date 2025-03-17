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

def evaluate_model(model, X_train, Y_train, X_test, Y_test, k) -> pd.DataFrame:
    metrics = {"K": [k]}
    for dataset, X, Y in [("train", X_train, Y_train), ("test", X_test, Y_test)]:
        Y_pred = model.predict(X)
        fpr, tpr, _ = roc_curve(Y, Y_pred)
        metrics[f"F1_Score_{dataset}"] = [f1_score(Y, Y_pred)]
        metrics[f"Recall_{dataset}"] = [recall_score(Y, Y_pred)]
        metrics[f"AUC_{dataset}"] = [auc(fpr, tpr)]
        metrics[f"Accuracy_{dataset}"] = [accuracy_score(Y, Y_pred)]
        metrics[f"FPR_{dataset}"] = [list(fpr)]
        metrics[f"TPR_{dataset}"] = [list(tpr)]
    return pd.DataFrame(metrics)

def search_params(model, params, X_train, Y_train, X_val, Y_val, model_name, max_combinations=np.inf, stop_iter=None, load_existing_model=True, save=True):
    models_results = load_json(RESULTS_PATH)

    if load_existing_model and model_name in models_results:
        return load_models_results(model_name)

    num_combinations = min(np.prod([len(v) for v in params.values()]), max_combinations)
    if num_combinations < 20:
        raise ValueError(f"O número de combinações ({num_combinations}) é menor que 20. Ajuste os hiperparâmetros.")

    X_train, Y_train, X_val, Y_val = map(np.array, [X_train, Y_train, X_val, Y_val])
    
    # Salva o uso de memória e o tempo de execução inicial
    initial_memory = memory_usage(-1, interval=0.1, max_usage=True)
    initial_time = time.time()

    # Cria o modelo RandomizedSearchCV com validação cruzada (cv=5)
    search_model = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        cv=5,
        n_iter=num_combinations,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # Realiza a busca com validação cruzada e mede a memória
    mem_usage, _ = memory_usage((search_model.fit, (X_train, Y_train)), retval=True, interval=0.1)
    mem_usage = [mem - initial_memory for mem in mem_usage]  # Subtrai a memória inicial para pegar a variação
    final_time = time.time() - initial_time
    
    # Se o modelo tem iteração (ex: MLP), coleta as métricas de cada época
    if hasattr(search_model.best_estimator_, 'loss_curve_'):
        df_iter = pd.DataFrame({
            'epoch': range(1, len(search_model.best_estimator_.loss_curve_) + 1),
            'loss': search_model.best_estimator_.loss_curve_,
            'accuracy': search_model.best_estimator_.validation_scores_
        })
    else:
        df_iter = None

    # Obtém as métricas finais da validação cruzada
    df_cv = evaluate_model(search_model.best_estimator_, X_train, Y_train, X_val, Y_val, 0)

    # Cria o dicionário de resultados
    results = {
        "model_name": model_name,
        "search_execution_time": final_time,
        "fit_execution_time": final_time,
        "memory_max": max(mem_usage),
        "memory_avg": np.mean(mem_usage),
        "best_params": search_model.best_params_,
        "model": f"./models/{model_name}.pkl",
        "result": f"./data/results/{model_name}.csv",
        "iter": None if df_iter is None else f"./data/results/{model_name}_iter.csv"
    }

    # Salva os resultados
    if save:
        save_model(search_model.best_estimator_, model_name)
        save_results(df_cv, model_name, df_iter)
        models_results[model_name] = results
        save_json(models_results, RESULTS_PATH)

    return df_cv, search_model.best_estimator_, results, df_iter
