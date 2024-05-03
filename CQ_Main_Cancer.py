import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from CQ_Cancer import QuadraticClassifier

        
if __name__ == "__main__":
    # Carregando o conjunto de dados Breast Cancer Wisconsin
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Dividindo os dados em conjuntos de treino e teste
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convertendo os rótulos para o formato exigido
    Y_treino = np.zeros((y_treino.size, 2))
    Y_teste = np.zeros((y_teste.size, 2))
    for i in range(y_treino.size):
        Y_treino[i, y_treino[i]] = 1
    for i in range(y_teste.size):
        Y_teste[i, y_teste[i]] = 1

    # Inicializando e treinando o classificador quadrático
    cq = QuadraticClassifier(X_treino.T, Y_treino.T, 2, 1)  # 2 classes
    cq.fit()

    # Testando o classificador
    accuracy, cm, sensibilidade, especificidade, f1 = cq.test(X_teste.T, Y_teste.T)
    print("matriz de confusão:")
    print(cm)
    print("Accuracy:", accuracy)
    print("sensibilidade:", sensibilidade)
    print("especificidade:", especificidade)
    print("f1:", f1)
    cq.plot_metrics_boxplot(accuracy, sensibilidade, especificidade, f1)
