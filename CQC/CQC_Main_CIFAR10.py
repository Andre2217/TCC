import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import cifar10  # type: ignore 
from CQC_CIFAR10 import QuadraticClassifier
import time
import cv2

def vetorizar_imagens(X):
    # Vetorizar cada imagem e concatenar em uma matriz
    X_vetorizado = np.reshape(X, (X.shape[0], -1))
    return X_vetorizado

def filtrar_classes_e_converter_para_cinza(X, y, classes):
    # Filtrar as imagens e rótulos para conter apenas as classes especificadas
    indices_filtrados = np.isin(y, classes).flatten()
    X_filtrado = X[indices_filtrados]
    y_filtrado = y[indices_filtrados]
    
    # Converter as imagens para escala de cinza
    X_cinza = np.array([cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) for imagem in X_filtrado])
    
    return X_cinza, y_filtrado

def calcular_metricas_matriz_confusao(matriz_confusao):
    TP = np.diag(matriz_confusao)
    FP = np.sum(matriz_confusao, axis=0) - TP
    FN = np.sum(matriz_confusao, axis=1) - TP
    TN = np.sum(matriz_confusao) - (TP + FP + FN)

    acuracia = (TP + TN) / np.sum(matriz_confusao)
    sensibilidade = TP / (TP + FN)
    especificidade = TN / (TN + FP)
    precisao = TP / (TP + FP)
    f1_score = 2 * (precisao * sensibilidade) / (precisao + sensibilidade)

    return acuracia, sensibilidade, especificidade, f1_score

if __name__ == "__main__":
    # Carregar a base de dados CIFAR-10
    (X_treino, y_treino), (X_teste, y_teste) = cifar10.load_data()

    # Selecionar apenas as classes 0 (avião) e 1 (automóvel)
    classes = [0, 1]
    # Filtrar e converter as imagens de treino
    X_treino_cinza, y_treino_filtrado = filtrar_classes_e_converter_para_cinza(X_treino, y_treino, classes)

    # Filtrar e converter as imagens de teste
    X_teste_cinza, y_teste_filtrado = filtrar_classes_e_converter_para_cinza(X_teste, y_teste, classes)
    
    # Vetorizar as imagens
    # X_treino_vetorizado = vetorizar_imagens(X_treino)
    # X_teste_vetorizado = vetorizar_imagens(X_teste)

    # Transformar as imagens em numpy arrays
    X_treino_cinza = np.array(X_treino_cinza)
    y_treino_filtrado = np.array(y_treino_filtrado)
    X_teste_cinza = np.array(X_teste_cinza)
    y_teste_filtrado = np.array(y_teste_filtrado)

     # Concatenar os vetores em uma matriz de dados
    X_treino_concatenado = X_treino_cinza.reshape(X_treino_cinza.shape[0], -1)
    X_teste_concatenado = X_teste_cinza.reshape(X_teste_cinza.shape[0], -1)

    # Convertendo os rótulos para o formato exigido (one-hot encoding)
    Y_treino = np.zeros((y_treino_filtrado.size, 2))
    Y_treino[np.arange(y_treino_filtrado.size), y_treino_filtrado.flatten()] = 1

    Y_teste = np.zeros((y_teste_filtrado.size, 2))
    Y_teste[np.arange(y_teste_filtrado.size), y_teste_filtrado.flatten()] = 1

    N_treino = X_treino_concatenado.shape[0]

    acuracias = []
    sensibilidades = []
    especificidades = []
    f1_scores = []
    matrizes_confusao = []
    tempo_rodada = []

    for i in range(100):
        seed = np.random.permutation(N_treino)
        X_treino_embaralhado = X_treino_concatenado[seed]
        Y_treino_embaralhado = Y_treino[seed]

        X_treino_split = X_treino_embaralhado[:int(N_treino * .9), :]
        Y_treino_split = Y_treino_embaralhado[:int(N_treino * .9), :]

        X_teste_split = X_treino_embaralhado[int(N_treino * .9):, :]
        Y_teste_split = Y_treino_embaralhado[int(N_treino * .9):, :]

        cq = QuadraticClassifier(X_treino_split.T, Y_treino_split.T, 2, 0.0001)
        
        inicio = time.time()
        cq.fit()
        fim = time.time()
        tempo = fim - inicio
        tempo_rodada.append(tempo)
        
        acuracia, matriz_confusao = cq.test(X_teste_split.T, Y_teste_split.T)
        acuracia, sensibilidade, especificidade, f1_score = calcular_metricas_matriz_confusao(matriz_confusao)

        acuracias.append(np.mean(acuracia))
        sensibilidades.append(np.mean(sensibilidade))
        especificidades.append(np.mean(especificidade))
        f1_scores.append(np.mean(f1_score))
        matrizes_confusao.append(matriz_confusao)

    # Encontrar os índices das melhores e piores rodadas
    indice_melhor = np.argmax(acuracias)
    indice_pior = np.argmin(acuracias)
    
    # Exibir as matrizes de confusão das melhores e piores rodadas
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(matrizes_confusao[indice_melhor], annot=True, fmt='.1f', cmap='coolwarm', ax=ax[0])
    ax[0].set_title(f'Melhor Matriz de Confusão (Acurácia: {acuracias[indice_melhor]:.5f})')
    ax[0].set_xlabel('Predito')
    ax[0].set_ylabel('Verdadeiro')

    sns.heatmap(matrizes_confusao[indice_pior], annot=True, fmt='.1f', cmap='coolwarm', ax=ax[1])
    ax[1].set_title(f'Pior Matriz de Confusão (Acurácia: {acuracias[indice_pior]:.5f})')
    ax[1].set_xlabel('Predito')
    ax[1].set_ylabel('Verdadeiro')

    plt.show()

    # Exibir tempo de treino
    tempo_medio = np.mean(tempo_rodada)
    print(f"Tempo médio de treino: {tempo_medio:.5f}")
    print(f"Maior tempo de um treinamento: {np.max(tempo_rodada):.5f}")
    print(f"Menor tempo de um treinamento: {np.min(tempo_rodada):.5f}")

    # Exibir as métricas das melhores e piores rodadas
    print(f"Melhor rodada (Rodada {indice_melhor + 1}):")
    print(f"Acurácia: {acuracias[indice_melhor]:.5f}")
    print(f"Sensibilidade: {sensibilidades[indice_melhor]:.5f}")
    print(f"Especificidade: {especificidades[indice_melhor]:.5f}")
    print(f"F1-score: {f1_scores[indice_melhor]:.5f}")

    print(f"\nPior rodada (Rodada {indice_pior + 1}):")
    print(f"Acurácia: {acuracias[indice_pior]:.5f}")
    print(f"Sensibilidade: {sensibilidades[indice_pior]:.5f}")
    print(f"Especificidade: {especificidades[indice_pior]:.5f}")
    print(f"F1-score: {f1_scores[indice_pior]:.5f}")

    # Plotar boxplot das métricas
    plt.figure(figsize=(10, 7))
    plt.boxplot([acuracias, sensibilidades, especificidades, f1_scores], labels=['Acurácia', 'Sensibilidade', 'Especificidade', 'F1-score'])
    plt.title('Boxplot das Métricas')
    plt.show()