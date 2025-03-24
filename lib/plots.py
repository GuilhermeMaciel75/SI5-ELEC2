import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def show_metrics(df_results, metrics=["F1_Score", "Recall", "AUC", "Accuracy"], model_name="", ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    df_results = df_results.sort_values(by="K")

    colors = {
        "F1_Score": ("darkblue", "lightblue"),
        "Recall": ("darkred", "lightcoral"),
        "AUC": ("darkgreen", "lightgreen"),
        "Accuracy": ("darkorange", "lightsalmon")
    }

    for metric in metrics:
        train_color, test_color = colors.get(metric, ("black", "gray"))
        ax.plot(df_results["K"], df_results[f"{metric}_train"], marker="o", linestyle="-", color=train_color, label=f"{metric} (train)")
        ax.plot(df_results["K"], df_results[f"{metric}_test"], marker="s", linestyle="--", color=test_color, label=f"{metric} (test)")

    ax.set_xlabel("K-fold")
    ax.set_ylabel("Valor da Métrica")
    ax.set_title(f"Métricas do Modelo {model_name} ao longo dos K folds")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.grid(True)

    if ax is None:
        return fig

def show_roc(df_results, ax=None):
    if "FPR_test" not in df_results.columns or "TPR_test" not in df_results.columns:
        raise KeyError("As colunas 'FPR_test' e 'TPR_test' não estão presentes no DataFrame.")

    best_model = df_results.iloc[0]
    fpr_best = np.array(best_model["FPR_test"])
    tpr_best = np.array(best_model["TPR_test"])

    if fpr_best is None or tpr_best is None or len(fpr_best) == 0 or len(tpr_best) == 0:
        raise ValueError("Os valores de 'FPR_test' e 'TPR_test' do melhor modelo são inválidos.")

    valid_fprs = [np.interp(np.linspace(0, 1, 100), df_results.iloc[i]["FPR_test"], df_results.iloc[i]["TPR_test"]) 
                  for i in range(len(df_results)) if df_results.iloc[i]["FPR_test"] is not None]

    if not valid_fprs:
        raise ValueError("Nenhum valor válido encontrado para 'TPR_test'.")

    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean(valid_fprs, axis=0)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr_best, tpr_best, label="Melhor Modelo", linestyle="-", color="darkblue", linewidth=2)
    ax.plot(mean_fpr, mean_tpr, label="Média dos Modelos", linestyle="--", color="darkred", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", label="Aleatório")

    ax.set_xlabel("Falso Positivo (FPR)")
    ax.set_ylabel("Verdadeiro Positivo (TPR)")
    ax.set_title("Curva ROC")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.grid()

    if ax is None:
        return fig

def show_bar_metrics(df_results, ax=None):
    metrics = ["F1_Score", "Recall", "AUC", "Accuracy"]
    train_metrics = [f"{m}_train" for m in metrics]
    test_metrics = [f"{m}_test" for m in metrics]

    df_means = df_results[train_metrics + test_metrics].mean()
    df_max = df_results[train_metrics + test_metrics].max()
    df_plot = pd.DataFrame({"Média": df_means, "Máximo": df_max})

    new_index = []
    for m in metrics:
        new_index.append(f"{m}_train")
        new_index.append(f"{m}_test")
    df_plot = df_plot.reindex(new_index)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ymin = df_plot.min().min() * 0.95
    ymax = 1.0
    ax.set_ylim(ymin, ymax)

    bars_means = ax.bar(df_plot.index, df_plot["Média"], color="blue", alpha=0.8)
    bars_max = ax.bar(df_plot.index, df_plot["Máximo"], color="red", alpha=0.6)

    ax.set_title("Comparação das métricas entre Média e Máximo")
    ax.set_ylabel("Valor da Métrica")
    ax.set_xticks(range(len(df_plot)))
    ax.set_xticklabels(df_plot.index, rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    if ax is None:
        return fig

def show_confusion_matrix(model, X_test, Y_test, ax=None):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    Y_pred = model.predict(X_test)
    cm = confusion_matrix(Y_test, Y_pred)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["DOWN", "UP"], yticklabels=["DOWN", "UP"], ax=ax)

    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusão")

    if ax is None:
        return fig

def model_evaluate(model, df_results, X_test, Y_test, model_name=""):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    show_metrics(df_results, model_name=model_name, ax=axes[0, 0])
    show_confusion_matrix(model, X_test, Y_test, ax=axes[0, 1])
    show_bar_metrics(df_results, ax=axes[1, 0])
    show_roc(df_results, ax=axes[1, 1])

    plt.tight_layout()
    plt.show()

def show_best_roc(df_results, ax=None):
    if "fpr" not in df_results.columns or "tpr" not in df_results.columns:
        raise KeyError("As colunas 'fpr' e 'tpr' não estão presentes no DataFrame.")

    # Encontrar o modelo com o melhor 'auc'
    best_model_idx = df_results['auc'].idxmax()  # Índice do modelo com o melhor 'auc'
    best_model = df_results.loc[best_model_idx]
    fpr_best = np.array(best_model["fpr"])
    tpr_best = np.array(best_model["tpr"])

    if fpr_best is None or tpr_best is None or len(fpr_best) == 0 or len(tpr_best) == 0:
        raise ValueError("Os valores de 'fpr' e 'tpr' do melhor modelo são inválidos.")

    # Calcular a curva ROC média
    valid_fprs = [
        np.interp(np.linspace(0, 1, 100), df_results.iloc[i]["fpr"], df_results.iloc[i]["tpr"])
        for i in range(len(df_results)) if df_results.iloc[i]["fpr"] is not None
    ]

    if not valid_fprs:
        raise ValueError("Nenhum valor válido encontrado para 'tpr'.")

    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean(valid_fprs, axis=0)

    # Plotando o gráfico
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr_best, tpr_best, label="Melhor Modelo (AUC Máximo)", linestyle="-", color="darkblue", linewidth=2)
    ax.plot(mean_fpr, mean_tpr, label="Média dos Modelos", linestyle="--", color="darkred", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", label="Aleatório")

    ax.set_xlabel("Falso Positivo (FPR)")
    ax.set_ylabel("Verdadeiro Positivo (TPR)")
    ax.set_title("Comparação: Melhor Modelo vs Média das Curvas ROC")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.grid()

    return fig

def show_bar_metrics2(df_results, ax=None):
    # Alterando para incluir as chaves corretas
    metrics = ["accuracy_train", "accuracy_test", "accuracy_val", "f1_score", "recall", "auc" ]

    # Verificando se todas as métricas estão no DataFrame
    missing_metrics = [metric for metric in metrics if metric not in df_results.columns]
    if missing_metrics:
        raise KeyError(f"As colunas {missing_metrics} não estão presentes no DataFrame.")

    # Calculando as médias e máximos
    df_means = df_results[metrics].mean()
    df_max = df_results[metrics].max()
    
    # Preparando o DataFrame para o gráfico
    df_plot = pd.DataFrame({"Média": df_means, "Máximo": df_max})

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Definindo o limite do gráfico para visualização
    ymin = df_plot.min().min() * 0.95
    ymax = 1.0
    ax.set_ylim(ymin, ymax)

    # Plotando as barras de média e máximo
    bars_means = ax.bar(df_plot.index, df_plot["Média"], color="blue", alpha=0.8)
    bars_max = ax.bar(df_plot.index, df_plot["Máximo"], color="red", alpha=0.6)

    # Configurando título, rótulos e grid
    ax.set_title("Comparação das Métricas: Média vs Máximo")
    ax.set_ylabel("Valor da Métrica")
    ax.set_xticks(range(len(df_plot)))
    ax.set_xticklabels(df_plot.index, rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    return fig

def show_metrics_comparison_line_plot(df_results, ax=None, y_min=0.8):
    # Definir as métricas a serem comparadas
    metrics = ["accuracy_train", "f1_score", "recall", "auc", "accuracy_train", "accuracy_val", "accuracy_test"]
    
    # Verificar se as métricas estão presentes nas colunas
    missing_metrics = [metric for metric in metrics if metric not in df_results.columns]
    if missing_metrics:
        raise KeyError(f"As colunas {missing_metrics} não estão presentes no DataFrame.")
    
    # Definir as cores pastéis para as linhas
    colors = ['#F59330FF', '#2B91F7FF', '#E651E6FF', '#321FDDFF', '#70C918FF', '#E64E4EFF', '#682892FF']
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    # Plotando cada métrica com uma linha diferente
    for i, metric in enumerate(metrics):
        ax.plot(df_results.index, df_results[metric], label=metric, color=colors[i], linewidth=2)
    
    # Definir o limite do eixo y entre 0.8 * min(metrics) e 1.0
    ymin = y_min
    ymax = 1.0
    ax.set_ylim(ymin, ymax)

    # Adicionar título e rótulos
    ax.set_title("Comparação das Métricas por Modelo")
    ax.set_xlabel("Modelos")
    ax.set_ylabel("Valor das Métricas")
    
    # Adicionar legenda
    ax.legend(loc="best")
    
    # Adicionar grid no eixo y
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    return fig

def plot_cv_performance(cv_results, title):
    # Extraindo as pontuações de treino e teste de cada split
    test_scores = [cv_results[f"split{i}_test_score"] for i in range(5)]
    train_scores = [cv_results[f"split{i}_train_score"] for i in range(5)]
    
    # Criando o gráfico
    plt.figure(figsize=(10, 6))
    
    # Plotando o desempenho de teste
    plt.plot(range(1, 6), test_scores, label='Test Score', marker='o', linestyle='-', color='blue')
    
    # Plotando o desempenho de treino
    plt.plot(range(1, 6), train_scores, label='Train Score', marker='o', linestyle='-', color='red')
    
    # Adicionando rótulos e título
    plt.xlabel('Split', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(title, fontsize=14)
    
    # Adicionando a legenda
    plt.legend()
    
    # Exibindo a grade
    plt.grid(True)
    
    # Exibindo o gráfico
    plt.show()

def plot_confusion_matrix(cm, title="Matriz de Confusão"):
    cm = np.array(cm)  # Garante que a matriz esteja no formato correto

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[str(i) for i in range(len(cm))],
                yticklabels=[str(i) for i in range(len(cm))])

    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title(title)
    plt.tight_layout()
    plt.show()



def plot_roc_curve(fpr, tpr, auc_score, title="Curva ROC"):
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {auc_score:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Aleatório')
    plt.xlabel("FPR (Taxa de Falsos Positivos)")
    plt.ylabel("TPR (Taxa de Verdadeiros Positivos)")
    plt.title(title)
    plt.legend()
    plt.show()

