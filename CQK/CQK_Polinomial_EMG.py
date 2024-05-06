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

        self.kernel_matrices = []
        self.mean_vectors = []
        self.pooled_kernel_matrix = np.zeros((self.N, self.N))
        self.kernel_matrices_inversion = []
        self.n = []
        self.determinant_kernel = []

    def fit(self):
        for c in range(self.c):
            idxs = self.Y_train[c, :] == 1
            x = self.X_train[:, idxs]
            n = x.shape[1]
            self.n.append(n)
            pi = n / self.N
            mu = np.mean(self.X_train[:, idxs], axis=1).reshape(self.p, 1)
            self.mean_vectors.append(mu)
            M = np.tile(mu, (1, n))
            kernel_matrix = self.polynomial_kernel(x, self.X_train, self.degree)
            self.kernel_matrices.append(kernel_matrix)
            self.pooled_kernel_matrix += pi * kernel_matrix

        for c in range(self.c):
            kernel_friedman = ((1 - self.lbd) * (self.n[c] * self.kernel_matrices[c]) + (self.lbd * self.N * self.pooled_kernel_matrix)) / ((1 - self.lbd) * self.n[c] + self.lbd * self.N)
            self.determinant_kernel.append(np.linalg.det(kernel_friedman))
            self.kernel_matrices_inversion.append(np.linalg.pinv(kernel_friedman))

    def test(self, X_test, Y_test):
        confusion_matrix = np.zeros((self.c, self.c))
        accuracy = 0
        for i in range(X_test.shape[1]):
            x_i = X_test[:, i].reshape(self.p, 1)
            gs = []
            for c in range(self.c):
                g = (x_i - self.mean_vectors[c]).T @ self.kernel_matrices_inversion[c] @ (x_i - self.mean_vectors[c])
                gs.append(g)
                if self.lbd != 1:
                    g = (-1 / 2 * (np.log(self.determinant_kernel[c]))) - (1 / 2 * (g))
            y_real = Y_test[:, i]
            y_pred = -np.ones(y_real.shape)
            y_pred[np.argmin(gs)] = 1

            idx_real, idx_predito = np.argmax(y_real), np.argmax(y_pred)
            if idx_real == idx_predito:
                accuracy += 1
            confusion_matrix[idx_predito, idx_real] += 1

        return accuracy / X_test.shape[1], confusion_matrix

    @staticmethod
    def polynomial_kernel(x, X, degree):
        return (1 + np.dot(x.T, X) ** degree)


# Exemplo de uso:
# Substitua os valores apropriados para X_train, Y_train, X_test, Y_test e outros parâmetros necessários.
# classifier = QuadraticClassifier(X_train, Y_train, c, lbd, gamma, degree)
# classifier.fit()
# accuracy, confusion_matrix = classifier.test(X_test, Y_test)
