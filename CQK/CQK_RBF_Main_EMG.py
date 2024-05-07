import numpy as np
import matplotlib.pyplot as plt
from CQK.CQK_RBF_EMG import QuadraticClassifier

if __name__ == "__main__":
    X = np.loadtxt("./EMG.csv", delimiter=',')
    colors = ['red', 'blue', 'magenta', 'purple', 'yellow']
    k = 0
    Y = np.empty((0, 5))
    
    # Gerar rótulos de classe
    for i in range(10):
        for j in range(5):
            y = -np.ones((1000, 5))
            y[:, j] = 1
            Y = np.concatenate((Y, y))
            k += 1000

    N, p = X.shape
    c = Y.shape[1]
    
    for i in range(100):
        seed = np.random.permutation(N)
        Xr = np.copy(X[seed, :])
        Yr = np.copy(Y[seed, :])

        # Divisão dos dados em treino e teste
        X_treino = Xr[:int(N * 0.8), :]
        Y_treino = Yr[:int(N * 0.8), :]
        X_teste = Xr[int(N * 0.8):, :]
        Y_teste = Yr[int(N * 0.8):, :]

        # Inicialização e ajuste do classificador
        cq = QuadraticClassifier(X_treino.T, Y_treino.T, 5, 1)
        cq.fit()

        # Teste do classificador
        accuracy, confusion_matrix = cq.test(X_teste.T, Y_teste.T)
        
        # Estatísticas
        means = np.mean(X_treino, axis=0)
        stds = np.std(X_treino, axis=0)
        max_values = np.max(X_treino, axis=0)
        min_values = np.min(X_treino, axis=0)
        
        # Imprimir resultados
        print("Acurácia:", accuracy)
        print("Matriz de Confusão:")
        print(confusion_matrix)
        print("Médias por classe:", means)
        print("Desvios padrão por classe:", stds)
        print("Maior valor por classe:", max_values)
        print("Menor valor por classe:", min_values)

        # Visualização opcional
        # plt.imshow(confusion_matrix, cmap='Blues')
        # plt.colorbar()
        # plt.show()
