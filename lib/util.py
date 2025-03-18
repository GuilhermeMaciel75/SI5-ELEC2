import os
import json
import pickle


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
    
def save_results(df_results, model_name, df_iter):
    results_path = f"./data/results/{model_name}.csv"
    df_results.to_csv(results_path, index=False)
    if df_iter is not None:
        iter_path = f"./data/results/{model_name}_iter.csv"
        df_iter.to_csv(iter_path, index)
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