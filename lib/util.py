import os
import json
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import pandas as pd


RANDOM_STATE = 51
RESULTS_PATH = "./data/results/models.json"
MODEL_PATH = "./models/"
RESULTS_DIR = "./data/results/"

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

def load_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def save_model(model, model_name):
    model_path = f"./models/{model_name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    return model_path

def load_model(model_name):
    model_path = os.path.join(MODEL_PATH, f"{model_name}.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None
    
def save_results(df_results, model_name):
    results_path = f"./data/results/{model_name}.csv"
    df_results.to_csv(results_path, index=False)
    return results_path

def show_results(results:dict):
    print("Resultados do Modelo: {}".format(results['model_name']))
    print()
    print("Tempo da Busca de Parâmetros: {:.3f}s".format(results['search_execution_time']))
    print("Melhores Parâmetros Encontrados: {}".format(results['best_params']))
    print()
    print("Tempo de Treinamento: {:.3f}s".format(results['fit_execution_time']))
    print("Média de Memória Utilizada: {:.2f} MB".format(results['memory_avg']))
    print("Pico de Memória Utilizada: {:.2f} MB".format(results['memory_max']))

def generate_markdown_report(best_model, results_list):
    # Inicializando o markdown
    markdown = f"### Resultados do Modelo\n\n"
    
    # Obter os melhores parâmetros do modelo
    best_params = best_model.get_params()
    markdown += f"**Melhores Parâmetros:**\n```\n{best_params}\n```\n\n"
    
    # Definindo as métricas a serem analisadas
    metrics = ["accuracy_train", "accuracy_val", "accuracy_test", "f1_score", "auc", "recall"]
    
    # Encontrar o modelo com o melhor AUC
    best_model_row = max(results_list, key=lambda x: x['auc'])  # Modelo com o maior AUC
    
    # Iterando sobre as métricas para calcular os valores
    markdown += "#### Resultados das Métricas:\n"
    for metric in metrics:
        # Obtendo os valores para cada métrica
        all_values = [model[metric] for model in results_list]
        
        # Calcular o melhor valor da métrica (máximo)
        best_value = max(all_values)
        
        # Calcular a média e desvio padrão da métrica
        avg_value = np.mean(all_values)
        std_value = np.std(all_values)
        
        # Encontrar o valor do modelo selecionado (o modelo com o melhor AUC)
        model_value = best_model_row[metric]
        
        # Adicionando os resultados ao markdown
        markdown += f"- **{metric.replace('_', ' ').capitalize()}**:\n"
        markdown += f"  - **Melhor valor geral**: {best_value:.4f}\n"
        markdown += f"  - **Média**: {avg_value:.4f}\n"
        markdown += f"  - **Desvio Padrão**: {std_value:.4f}\n"
        markdown += f"  - **Valor no Modelo Selecionado**: {model_value:.4f}\n\n"

    return markdown

def get_model_metrics(model, model_name, X_train, Y_train, X_val, Y_val, X_test, Y_test):
    # Avaliação dos modelos e adição ao DataFrame
    print(f"🔍 Avaliando modelo {model_name}...")

    # Previsões
    Y_pred_train = model.predict(X_train)
    Y_pred_val = model.predict(X_val)
    Y_pred_test = model.predict(X_test)

    # Probabilidades para AUC-ROC
    Y_prob_train = model.predict_proba(X_train)[:, 1]
    Y_prob_val = model.predict_proba(X_val)[:, 1]
    Y_prob_test = model.predict_proba(X_test)[:, 1]

    # Curva ROC
    fpr_train, tpr_train, _ = roc_curve(Y_train, Y_prob_train)
    fpr_val, tpr_val, _ = roc_curve(Y_val, Y_prob_val)
    fpr_test, tpr_test, _ = roc_curve(Y_test, Y_prob_test)

    # Matriz de Confusão
    cm_train = confusion_matrix(Y_train, Y_pred_train)
    cm_val = confusion_matrix(Y_val, Y_pred_val)
    cm_test = confusion_matrix(Y_test, Y_pred_test)

    # Salvando resultados no DataFrame
    df_results = pd.DataFrame([
        {
            "Data": "Train",
            "Accuracy": accuracy_score(Y_train, Y_pred_train),
            "F1_Score": f1_score(Y_train, Y_pred_train),
            "Recall": recall_score(Y_train, Y_pred_train),
            "Precision": precision_score(Y_train, Y_pred_train),
            "AUC_ROC": roc_auc_score(Y_train, Y_prob_train),
            "Confusion_Matrix": [cm_train],
            "FPR": [fpr_train],
            "TPR": [tpr_train]
        },
        {
            "Data": "Validation",
            "Accuracy": accuracy_score(Y_val, Y_pred_val),
            "F1_Score": f1_score(Y_val, Y_pred_val),
            "Recall": recall_score(Y_val, Y_pred_val),
            "Precision": precision_score(Y_val, Y_pred_val),
            "AUC_ROC": roc_auc_score(Y_val, Y_prob_val),
            "Confusion_Matrix": [cm_val],
            "FPR": [fpr_val],
            "TPR": [tpr_val],
        },
        {
            "Data": "Test",
            "Accuracy": accuracy_score(Y_test, Y_pred_test),
            "F1_Score": f1_score(Y_test, Y_pred_test),
            "Recall": recall_score(Y_test, Y_pred_test),
            "Precision": precision_score(Y_test, Y_pred_test),
            "AUC_ROC": roc_auc_score(Y_test, Y_prob_test),
            "Confusion_Matrix": [cm_test],
            "FPR": [fpr_test],
            "TPR": [tpr_test]
        }
    ])

    return df_results