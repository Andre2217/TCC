import numpy as np
import matplotlib.pyplot as plt

class QuadraticClassifier:

    def __init__(self,X_train,Y_train,c,lbd=0) -> None:
        self.X_train = X_train
        self.Y_train = Y_train
        self.c = c
        self.p,self.N = self.X_train.shape
        self.lbd = lbd
        
        self.covariances = []
        self.friedman_covariances = []
        self.mean_vectors = []
        self.pooled_covariance = np.zeros((self.p,self.p))
        self.covariances_invertion = []
        self.n = []
        self.determinant_cov = []
        

    def fit(self):
        
            for c in range(self.c):
                idxs = self.Y_train[c, :] == 1
                x = self.X_train[:, idxs]
                n = x.shape[1]


                if n == 0:
                    self.n.append(0)
                    continue
                self.n.append(n)
                pi = n / self.N
                mu = np.mean(self.X_train[:, idxs], axis=1).reshape(self.p, 1)
                self.mean_vectors.append(mu)
                M = np.tile(mu, (1, n))
                cov = 1 / n * ((x - M) @ (x - M).T)
                self.covariances.append(cov)
                self.pooled_covariance += pi * cov

                cov_friedman = ((1 - self.lbd) * (self.n[c] * self.covariances[c]) + (self.lbd * self.N * self.pooled_covariance)) / ((1 - self.lbd) * self.n[c] + self.lbd * self.N)
                self.determinant_cov.append(np.linalg.det(cov_friedman))
                self.covariances_invertion.append(np.linalg.pinv(cov_friedman))



    def test(self, X_test, Y_test):
        confusion_matrix = np.zeros((self.c, self.c))
        accuracy = 0
        total_samples = X_test.shape[1]
        
        for i in range(total_samples):
            x_i = X_test[:, i].reshape(self.p, 1)
            gs = []
            for c in range(self.c):
                if c >= len(self.mean_vectors) or c >= len(self.covariances_invertion):
                    print(f"Index out of range: c={c}")
                    continue
                g = (x_i - self.mean_vectors[c]).T @ self.covariances_invertion[c] @ (x_i - self.mean_vectors[c])
                gs.append(g)
            y_real = Y_test[:, i]
            y_pred = -np.ones(y_real.shape)
            y_pred[np.argmin(gs)] = 1

            idx_real, idx_predito = np.argmax(y_real), np.argmax(y_pred)
            if idx_real == idx_predito:
                accuracy += 1
            confusion_matrix[idx_predito, idx_real] += 1
        
        accuracy /= total_samples
    
        return accuracy, confusion_matrix


    
    # def plot_metrics_boxplot(self, accuracy, sensitivity, specificity, f_score):
    #     metrics_data = [accuracy, sensitivity, specificity, f_score]
    #     metrics_labels = ['Accuracy', 'Sensitivity', 'Specificity', 'F1-score']

    #     plt.figure(figsize=(10, 6))
    #     plt.boxplot(metrics_data)
    #     plt.xticks(range(1, len(metrics_labels) + 1), metrics_labels)
    #     plt.title('Boxplot of Metrics')
    #     plt.ylabel('Value')
    #     plt.grid(True)
    #     plt.show()
