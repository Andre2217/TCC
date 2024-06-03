import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from CQK_RBF_EMG import QuadraticClassifier

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

    Dataset = np.loadtxt("./EMGDataset.csv", delimiter=',')
    DadosClasse1 = Dataset[Dataset[:,2]==4,:]
    DadosClasse2 = Dataset[Dataset[:,2]==5,:]
    
    dados_filtrados = np.vstack((DadosClasse1, DadosClasse2))

    # Separa em X e Y
    X = dados_filtrados[:, :2]  # As duas primeiras colunas
    Y = dados_filtrados[:, 2]   # A terceira coluna
    
    # Transformar Y em uma matriz binária (one-hot encoding)
    Y_binario = np.zeros((Y.size, 2))
    Y_binario[Y == 4, 0] = 1
    Y_binario[Y == 5, 1] = 1

    N, p = X.shape
    c = Y_binario.shape[1]

    acuracias = []
    sensibilidades = []
    especificidades = []
    f1_scores = []
    matrizes_confusao = []
    tempo_rodada = []

    for i in range(100):
        seed = np.random.permutation(N)
        Xr = np.copy(X[seed, :])
        Yr = np.copy(Y_binario[seed])

        X_treino = Xr[0:int(N * .8), :]
        Y_treino = Yr[0:int(N * .8), :]

        X_teste = Xr[int(N * .8):, :]
        Y_teste = Yr[int(N * .8):, :]

        cq = QuadraticClassifier(X_treino.T, Y_treino.T, c, 1)
        
        inicio = time.time()
        cq.fit()
        fim = time.time()
        tempo = fim - inicio
        tempo_rodada.append(tempo)
        
        acc, matriz_confusao = cq.test(X_teste.T, Y_teste.T)
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
