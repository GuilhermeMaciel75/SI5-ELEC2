import os
import json
import pickle
import numpy as np

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
