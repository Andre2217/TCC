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
            idxs = self.Y_train[c,:] == 1
            x = self.X_train[:,idxs]
            n = x.shape[1]
            self.n.append(n)
            pi = n/self.N
            mu = np.mean(self.X_train[:,idxs],axis=1).reshape(self.p,1)  
            self.mean_vectors.append(mu)
            M = np.tile(mu,(1,n))
            cov = 1/n * ((x-M)@(x-M).T)
            self.covariances.append(cov)          
            self.pooled_covariance += pi*cov

        for c in range(self.c):
            cov_friedman = ((1-self.lbd)*(self.n[c]*self.covariances[c]) + (self.lbd*self.N*self.pooled_covariance))/((1-self.lbd)*self.n[c] + self.lbd*self.N)
            self.determinant_cov.append(np.linalg.det(cov_friedman))
            self.covariances_invertion.append(np.linalg.pinv(cov_friedman))
            
            


    def test(self,X_test,Y_test):
        confusion_matrix = np.zeros((self.c,self.c))
        accuracy = 0
        sensitivity = 0
        specificity = 0
        f_score = 0
        total_samples = X_test.shape[1]
        
        for i in range(total_samples):
            x_i = X_test[:,i].reshape(self.p,1)
            gs = []
            for c in range(self.c):
                g = (x_i-self.mean_vectors[c]).T@self.covariances_invertion[c]@(x_i-self.mean_vectors[c])
                gs.append(g)
                if(self.lbd != 1):
                    g = (-1/2*(np.log(self.determinant_cov[c])))-(1/2*(g))
            y_real = Y_test[:,i]
            y_pred = -np.ones(y_real.shape)
            y_pred[np.argmin(gs)] = 1

            idx_real,idx_predito = np.argmax(y_real),np.argmax(y_pred)
            if idx_real==idx_predito:
                accuracy+=1
            confusion_matrix[idx_predito,idx_real]+=1
        
        accuracy /= total_samples
        
        true_positive = confusion_matrix[1, 1]
        false_positive = confusion_matrix[1, 0]
        false_negative = confusion_matrix[0, 1]
        true_negative = confusion_matrix[0, 0]

        sensitivity = true_positive / (true_positive + false_negative)
        specificity = true_negative / (true_negative + false_positive)
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f_score = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

        return accuracy, confusion_matrix, sensitivity, specificity, f_score
    
    
    
    
    def plot_metrics_boxplot(self, accuracy, sensitivity, specificity, f_score):
        metrics_data = [accuracy, sensitivity, specificity, f_score]
        metrics_labels = ['Accuracy', 'Sensitivity', 'Specificity', 'F1-score']

        plt.figure(figsize=(10, 6))
        plt.boxplot(metrics_data)
        plt.xticks(range(1, len(metrics_labels) + 1), metrics_labels)
        plt.title('Boxplot of Metrics')
        plt.ylabel('Value')
        plt.grid(True)
        plt.show()

