import numpy as np
import matplotlib.pyplot as plt

class QuadraticClassifier:

    def __init__(self, X_train, Y_train, c, lbd=0, degree=2):

        self.X_train = X_train
        self.Y_train = Y_train
        self.c = c
        self.p, self.N = self.X_train.shape
        self.lbd = lbd
        self.degree = degree

        self.kernel_matrices = []  # Lista para armazenar as matrizes de kernel de cada classe
        self.mean_vectors = []  # Lista para armazenar os vetores médios de cada classe
        self.pooled_kernel_matrix = np.zeros((self.N, self.N))  # Matriz de kernel combinada
        self.kernel_matrices_inversion = []  # Lista para armazenar as inversões das matrizes de kernel de cada classe
        self.n = []  # Lista para armazenar o número de amostras de cada classe
        self.determinant_kernel = []  # Lista para armazenar os determinantes das matrizes de kernel de cada classe

    def fit(self):
        """
        Ajusta o classificador aos dados de treinamento.
        """
        for c in range(self.c):
            idxs = self.Y_train[c, :] == 1  # Obtém os índices das amostras pertencentes à classe c
            x = self.X_train[:, idxs]  # Obtém as amostras de treinamento da classe c
            n = x.shape[1]  # Calcula o número de amostras da classe c
            self.n.append(n)
            pi = n / self.N  # Calcula a proporção de amostras da classe c
            mu = np.mean(self.X_train[:, idxs], axis=1).reshape(self.p, 1)  # Calcula o vetor médio da classe c
            self.mean_vectors.append(mu)
            M = np.tile(mu, (1, n))  # Repete o vetor médio para formar uma matriz
            kernel_matrix = self.polynomial_kernel(x, self.X_train, self.degree)  # Calcula a matriz de kernel
            self.kernel_matrices.append(kernel_matrix)  # Adiciona a matriz de kernel à lista
            self.pooled_kernel_matrix += pi * kernel_matrix  # Atualiza a matriz de kernel combinada

        for c in range(self.c):
            # Calcula a matriz de kernel de Friedman
            kernel_friedman = ((1 - self.lbd) * (self.n[c] * self.kernel_matrices[c]) + 
                               (self.lbd * self.N * self.pooled_kernel_matrix)) / ((1 - self.lbd) * self.n[c] + self.lbd * self.N)
            self.determinant_kernel.append(np.linalg.det(kernel_friedman))  # Calcula e armazena o determinante da matriz de kernel
            self.kernel_matrices_inversion.append(np.linalg.pinv(kernel_friedman))  # Calcula e armazena a inversão da matriz de kernel

    def test(self, X_test, Y_test):

        confusion_matrix = np.zeros((self.c, self.c))  # Inicializa a matriz de confusão
        accuracy = 0
        for i in range(X_test.shape[1]):  # Loop sobre as amostras de teste
            x_i = X_test[:, i].reshape(self.p, 1)  # Obtém a i-ésima amostra de teste
            gs = []
            for c in range(self.c):  # Loop sobre as classes
                # Calcula a função discriminante quadrática
                g = (x_i - self.mean_vectors[c]).T @ self.kernel_matrices_inversion[c] @ (x_i - self.mean_vectors[c])
                gs.append(g)
                if self.lbd != 1:
                    g = (-1 / 2 * (np.log(self.determinant_kernel[c]))) - (1 / 2 * (g))
            y_real = Y_test[:, i]  # Obtém os rótulos reais da i-ésima amostra
            y_pred = -np.ones(y_real.shape)  # Inicializa os rótulos preditos como -1
            y_pred[np.argmin(gs)] = 1  # Atribui o rótulo 1 à classe com menor valor de função discriminante

            idx_real, idx_predito = np.argmax(y_real), np.argmax(y_pred)  # Obtém os índices do rótulo real e predito
            if idx_real == idx_predito:  # Verifica se a predição foi correta
                accuracy += 1
            confusion_matrix[idx_predito, idx_real] += 1  # Atualiza a matriz de confusão

        return accuracy / X_test.shape[1], confusion_matrix  # Retorna a acurácia e a matriz de confusão

    @staticmethod
    def polynomial_kernel(x, X, degree):

        return (2*x*X + ((x**2)*(X**2))+1)
