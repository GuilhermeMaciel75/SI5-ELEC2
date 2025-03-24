from sklearn.metrics import accuracy_score, f1_score, roc_curve, roc_auc_score, recall_score
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from tqdm.auto import tqdm
from lib.util import (
    RANDOM_STATE, RESULTS_PATH, RESULTS_DIR, 
    load_json, save_json, save_model, load_model, save_results
)
import os
import pickle

# Função para avaliar o modelo com base em várias métricas no conjunto de validação
def evaluate_model_metrics(model, X_val, Y_val, test=False):
    Y_pred = model.predict(X_val)
    
    metrics = {
        'accuracy': accuracy_score(Y_val, Y_pred),
        'f1_score': f1_score(Y_val, Y_pred),
        'recall': recall_score(Y_val, Y_pred),
        'auc': roc_auc_score(Y_val, Y_pred),
    }
    
    if test:
        fpr, tpr, _ = roc_curve(Y_val, Y_pred)
        metrics['fpr'] = list(fpr),
        metrics['tpr'] = list(tpr)
    
    return metrics


def cv_best_model(model,params,  X_val, Y_val):
    kfold = StratifiedKFold(n_splits=5, random_state=RANDOM_STATE, shuffle=True)
    
    cv_metrics = []
    
    for fold, (train_idx, val_idx) in tqdm(enumerate(kfold.split(X_val, Y_val), 1), desc=f"Realizando CV dos parâmetros {params}"):
        X_train_fold, X_val_fold = X_val[train_idx], X_val[val_idx]
        Y_train_fold, Y_val_fold = Y_val[train_idx], Y_val[val_idx]
        
        # Treina o modelo
        model.set_params(**params)
        model.fit(X_train_fold, Y_train_fold)
        
        # Avalia o modelo no conjunto de validação do fold
        fold_metrics = evaluate_model_metrics(model, X_val_fold, Y_val_fold)
        fold_metrics['fold'] = fold
        cv_metrics.append(fold_metrics)
    
    return cv_metrics

def search_params(model, params, model_name, dataset):
    
    X_train, Y_train, X_val, Y_val, X_test, Y_test = map(np.array, dataset)
    
    best_params = {}    
    find_models  = 0
    
    MODEL_PATH = f'./models/{model_name}'
    
    for i in tqdm(range(20), desc="Realizando a Busca de Parâmetros por 20 iteraçòes"):
        search_model = RandomizedSearchCV(
            estimator=model,
            param_distributions=params,
            cv=5,
            scoring='roc_auc',
            n_iter=20,
            random_state=RANDOM_STATE + i,
            n_jobs=-1,
            refit=True
        )
        
        search_model.fit(X_train, Y_train)
        
        # Se o mesmo conjunto de parâmetros foi escolhiodo anteriormente ignora
        if (params_str :=str(search_model.best_params_)) in best_params:
            continue
        
        find_models += 1
                
        model_data = {
            "params" : search_model.best_params_,
            "score" : search_model.best_score_,
            "model_time" : search_model.refit_time_,
            "train_score" : evaluate_model_metrics(search_model.best_estimator_, X_test, Y_test,True),
            "validation_matrix" : cv_best_model(model, search_model.best_params_, X_val, Y_val),
            "model_path" : f"./models/{model_name}/{find_models}.pkl"
        }
        
        best_params[params_str] = model_data
        
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)
        
        with open(f"{MODEL_PATH}/{find_models}.pkl", 'wb') as file:
            pickle.dump(search_model.best_estimator_, file)
    
    save_json(best_params, f"{MODEL_PATH}/{model_name}.json")
    
    return best_params
        
        

def search_paramsv2(model, params, model_name, dataset, n_iter=20, score='auc', verbose=0, save=True):
    
    MODEL_PATH = f'./models/{model_name}'
    
    if os.path.exists(MODEL_PATH) and not save:
        best_params = load_json(f"{MODEL_PATH}/{model_name}.json")

        with open(f"{MODEL_PATH}/{model_name}.pkl", 'wb') as file:
            best_model = pickle.load(file)
        
        best_cv = pd.read_csv(f"{MODEL_PATH}/best_cv.csv")
        mean_scores = pd.read_csv(f"{MODEL_PATH}/mean_scores.csv")
        
        return best_params, best_model, best_cv, None, mean_scores
        
    
    X_train, Y_train, X_val, Y_val, X_test, Y_test = map(np.array, dataset)
    
    best_params = {}
    best_model = None
    best_score = -float('inf')
    best_cv = None
    all_cv = pd.DataFrame()
    
    
    for i in tqdm(range(20), desc="Realizando a Busca de Parâmetros por 20 iterações"):
        
        search_model = RandomizedSearchCV(
            estimator=model,
            param_distributions=params,
            cv=5,
            scoring='roc_auc',
            n_iter=n_iter,
            random_state=RANDOM_STATE + i,
            n_jobs=-1,
            refit=True,
            return_train_score=True,
            verbose=verbose
        )
        
        search_model.fit(X_train, Y_train)
        
        all_cv = pd.concat([all_cv, pd.DataFrame(search_model.cv_results_).iloc[search_model.best_index_]], axis=1)
                
        Y_pred_train = search_model.best_estimator_.predict(X_train)
        Y_pred_val = search_model.best_estimator_.predict(X_val)
        Y_pred_test = search_model.best_estimator_.predict(X_test)
        
        fpr, tpr, _ = roc_curve(Y_test, Y_pred_test)
            
        best_params[i] = {
            "params" : search_model.best_params_,
            "score" : search_model.best_score_,
            "model_time" : search_model.refit_time_,
            
            'accuracy_train': accuracy_score(Y_train, Y_pred_train),
            'accuracy_val' : accuracy_score(Y_val, Y_pred_val),
            'accuracy_test': accuracy_score(Y_test, Y_pred_test),
            
            'f1_score': f1_score(Y_test, Y_pred_test),
            'recall': recall_score(Y_test, Y_pred_test),
            'auc': roc_auc_score(Y_test, Y_pred_test),
            
            'fpr': list(fpr),
            'tpr': list(tpr),
            
            "model_path" : f"./models/{model_name}/{i}.pkl"
        }
        
        if best_params[i].get(score) > best_score:
            best_score = best_params[i].get(score)
            best_model = search_model.best_estimator_
            best_cv = pd.DataFrame(search_model.cv_results_).iloc[search_model.best_index_]
            best_loss = search_model.best_estimator_.loss_curve_ \
                if hasattr(search_model.best_estimator_, "loss_curve_") else None

        
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)
        
    
    with open(f"{MODEL_PATH}/{model_name}.pkl", 'wb') as file:
            pickle.dump(search_model.best_estimator_, file)
    
    best_cv.to_csv(f"{MODEL_PATH}/best_cv.csv")
    
    save_json(best_params, f"{MODEL_PATH}/{model_name}.json")
    
    rows_to_select = [row for row in all_cv.index if 'split' in row and 'score' in row]
    subset_cv = all_cv.loc[rows_to_select]
    subset_cv
    mean_scores = subset_cv.mean(axis=1)
    mean_scores.to_csv(f"{MODEL_PATH}/mean_scores.csv")
        
    return best_params, best_model, best_cv, best_loss, mean_scores