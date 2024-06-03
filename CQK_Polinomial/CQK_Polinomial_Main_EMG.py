import numpy as np
import matplotlib.pyplot as plt
from CQK_Polinomial_EMG import QuadraticClassifier  # Importa a classe QuadraticClassifier do módulo

if __name__ == "__main__":
    # Carrega os dados do arquivo EMG.csv
    X = np.loadtxt("./EMG.csv", delimiter=',')
    colors = ['red', 'blue', 'magenta', 'purple', 'yellow']
    k = 0
    Y = np.empty((0, 5))  # Inicializa uma matriz vazia para armazenar os rótulos das classes
    
    # Gera rótulos de classe
    for i in range(10):
        for j in range(5):
            y = -np.ones((1000, 5))
            y[:, j] = 1
            Y = np.concatenate((Y, y))  # Concatena os rótulos das classes à matriz Y
            k += 1000

    N, p = X.shape  # Obtém o número de amostras e o número de features
    c = Y.shape[1]  # Obtém o número de classes

    # Seleciona aleatoriamente 1000 amostras
    random_indices = np.random.choice(N, 1000, replace=False)
    X = X[random_indices]
    Y = Y[random_indices]

    for i in range(100):  # Loop para repetir o processo de treinamento e teste várias vezes
        seed = np.random.permutation(1000)  # Permuta os índices das amostras
        Xr = np.copy(X[seed, :])  # Aplica a permutação aos dados de entrada
        Yr = np.copy(Y[seed, :])  # Aplica a permutação aos rótulos das classes

        # Divide os dados em conjunto de treinamento e teste
        X_treino = Xr[:int(1000 * 0.8), :]
        Y_treino = Yr[:int(1000 * 0.8), :]
        X_teste = Xr[int(1000 * 0.8):, :]
        Y_teste = Yr[int(1000 * 0.8):, :]

        # Inicialização e ajuste do classificador
        cq = QuadraticClassifier(X_treino.T, Y_treino.T, 5, lbd=1, degree=2)  # Inicializa o classificador
        cq.fit()  # Ajusta o classificador aos dados de treinamento

        # Teste do classificador
        accuracy, confusion_matrix = cq.test(X_teste.T, Y_teste.T)  # Testa o classificador nos dados de teste
        
        # Estatísticas dos dados de treinamento
        means = np.mean(X_treino, axis=0)  
        stds = np.std(X_treino, axis=0)  
        max_values = np.max(X_treino, axis=0) 
        min_values = np.min(X_treino, axis=0)  
        
        # Imprime os resultados
        print("Acurácia:", accuracy)
        print("Matriz de Confusão:")
        print(confusion_matrix)
        print("Médias por classe:", means)
        print("Desvios padrão por classe:", stds)
        print("Maior valor por classe:", max_values)
        print("Menor valor por classe:", min_values)
