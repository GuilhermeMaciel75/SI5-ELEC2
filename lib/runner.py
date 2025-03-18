from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, recall_score
import numpy as np
import time
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from tqdm.auto import tqdm
from lib.util import (
    RANDOM_STATE, RESULTS_PATH, RESULTS_DIR, 
    load_json, save_json, save_model, load_model, save_results
)

# Função para avaliar o modelo com base em várias métricas no conjunto de validação
def evaluate_model_metrics(model, X_val, Y_val):
    Y_pred = model.predict(X_val)
    
    fpr, tpr, _ = roc_curve(Y_val, Y_pred)
    
    metrics = {
        'accuracy': accuracy_score(Y_val, Y_pred),
        'f1_score': f1_score(Y_val, Y_pred),
        'recall': recall_score(Y_val, Y_pred),
        'auc': auc(fpr, tpr),
        'fpr': list(fpr),
        'tpr': list(tpr),
    }
    
    return metrics

# Função para selecionar o melhor modelo com base nas métricas
def select_best_model(models_metrics):
    best_score = -np.inf
    best_model = None
    best_params = None
    best_param_time = None

    for param, value in models_metrics.items():
        metrics = value['metrics']
        
        score = metrics['f1_score'] + metrics['accuracy'] - np.std(metrics['accuracy'])
        
        if score > best_score:
            best_score = score
            best_model = value['model']
            best_params = metrics['param']
            best_param_time = metrics['train_time']
    
    return best_model, best_score, best_param_time, best_params


# Função principal que realiza a busca de parâmetros e validação
def new_search_params(model, params, model_name,
                      X_train, Y_train, X_val, Y_val, X_test, Y_test,
                      std_w = 10, f1_w=1, acc_w=1,
                      max_combinations=20, top_n=5,
                      load_existing_model=True, save=True):
    
    models_results = load_json(RESULTS_PATH)

    if load_existing_model and model_name in models_results:
        return load_models_results(model_name)

    num_combinations = min(np.prod([len(v) for v in params.values()]), max_combinations)
    if num_combinations < 20:
        raise ValueError(f"O número de combinações ({num_combinations}) é menor que 20. Ajuste os hiperparâmetros.")

    X_train, Y_train, X_val, Y_val, X_test, Y_test = map(np.array, [X_train, Y_train, X_val, Y_val, X_test, Y_test])

    # Função de scoring personalizada
    def custom_scoring(estimator, X, y):
        """Função de scoring personalizada que combina acurácia, f1_score e o desvio padrão da acurácia."""

        Y_pred = estimator.predict(X)

        accuracy = accuracy_score(y, Y_pred)
        f1 = f1_score(y, Y_pred)
        acc_std = np.std(Y_pred)

        score = (f1_w*f1 + accuracy*acc_w)/(f1_w + acc_w) - (acc_std*std_w)
        return score
    
    search_model_initial_time = time.time()

    # Realiza a busca com RandomizedSearchCV
    search_model = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        cv=5,
        scoring=custom_scoring,  # Usando a função de scoring personalizada definida acima
        n_iter=num_combinations,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        return_train_score=True,  # Para retornar a avaliação feita em cada k-fold durante a CV
    )
    
    search_model.fit(X_train, Y_train)
    search_model_final_time = time.time() - search_model_initial_time

    models_metrics = {}

    # Obter o rank de todos os modelos e s eleciona os top_n modelos com base no rank_test_score
    ranks = search_model.cv_results_['rank_test_score']
    top_n_indices = np.argsort(ranks)[:top_n]
    
    # Avaliar os modelos com os melhores parâmetros
    for idx in tqdm(top_n_indices, desc=f"Avaliando os top {top_n} modelos com a Validação"):
        param = search_model.cv_results_["params"][idx]
        model_val = model.set_params(**param)
        init_time = time.time()
        model_val.fit(X_train, Y_train)
        load_time = time.time() - init_time

        # Avalia o modelo no conjunto de validação
        metrics = evaluate_model_metrics(model_val, X_val, Y_val)
        metrics['train_time'] = load_time
        metrics['param'] = param


        models_metrics[str(param)] = {'model': model_val, 'metrics': metrics} 

    # Selecionar o melhor modelo com base nas métricas
    best_model, score, best_time, best_params = select_best_model(models_metrics)

    # Avaliar o melhor modelo no conjunto de teste
    test_metrics = evaluate_model_metrics(best_model, X_test, Y_test)

    # Realizar a validação cruzada para o melhor modelo encontrado
    kfold = StratifiedKFold(n_splits=5, random_state=RANDOM_STATE, shuffle=True)
    cv_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, Y_train), 1):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        Y_train_fold, Y_val_fold = Y_train[train_idx], Y_train[val_idx]
        
        # Treina o modelo
        best_model.fit(X_train_fold, Y_train_fold)
        
        # Avalia o modelo no conjunto de validação do fold
        fold_metrics = evaluate_model_metrics(best_model, X_val_fold, Y_val_fold)
        fold_metrics['fold'] = fold
        cv_metrics.append(fold_metrics)
    
    df_cvs = pd.DataFrame(cv_metrics)
    
    # Criando o df ao longo de iterações para MLP e similiares
    df_iter = pd.DataFrame({
        'epoch': range(len(search_model.best_estimator_.loss_curve_)),
        'loss': search_model.best_estimator_.loss_curve_,
        'accuracy': search_model.best_estimator_.validation_scores_
    }) if hasattr(search_model.best_estimator_, 'loss_curve_') else None
    
    # Calculando as métricas médias e desvio padrão durante o treinamento e validação
    train_avg_accuracy = np.mean(search_model.cv_results_['mean_train_score'][search_model.cv_results_['param_' + list(best_params.keys())[0]] == best_params[list(best_params.keys())[0]]])
    train_avg_accuracy_std = np.std(search_model.cv_results_['mean_train_score'][search_model.cv_results_['param_' + list(best_params.keys())[0]] == best_params[list(best_params.keys())[0]]])
    train_avg_score = np.mean(search_model.cv_results_['mean_test_score'][search_model.cv_results_['param_' + list(best_params.keys())[0]] == best_params[list(best_params.keys())[0]]])
    val_avg_accuracy = np.mean(df_cvs['accuracy'])
    val_avg_f1_score = np.mean(df_cvs['f1_score'])
    val_avg_recall = np.mean(df_cvs['recall'])

    # Dicionário de resultados
    results = {
        "model_name": model_name,
        "search_execution_time": search_model_final_time,
        "best_model_train_time": best_time,
        "best_params": best_params,
        "best_score": score,
        'train_avg_accuracy': train_avg_accuracy,
        'train_avg_accuracy_std': train_avg_accuracy_std,
        'train_avg_score': train_avg_score,
        'val_avg_accuracy': val_avg_accuracy,
        'val_avg_f1_score': val_avg_f1_score,
        'val_avg_recall': val_avg_recall,
        "test_accuracy": test_metrics['accuracy'],
        "test_f1_score": test_metrics['f1_score'],
        "test_recall": test_metrics['recall'],
        "test_auc": test_metrics['auc'],
        "fpr": test_metrics['fpr'],
        "tpr": test_metrics['tpr'],
        "model": f"./models/{model_name}.pkl",
        "result": f"./data/results/{model_name}.csv",
    }

    # Salva os resultados
    if save:
        save_model(best_model, model_name)
        save_results(df_cvs, model_name)
        models_results[model_name] = results
        save_json(models_results, RESULTS_PATH)

    return df_cvs, best_model, results, df_iter
