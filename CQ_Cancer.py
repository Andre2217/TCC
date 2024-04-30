import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

# Função para calcular as métricas e plotar os boxplots
def calculate_metrics_and_boxplots(y_true, y_pred):
    # Calculando a matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculando a acurácia
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculando a sensibilidade (recall) e a especificidade
    sensitivity = recall_score(y_true, y_pred)
    specificity = recall_score(y_true, y_pred, pos_label=0)
    
    # Calculando o F-score
    fscore = f1_score(y_true, y_pred)
    
    return cm, accuracy, sensitivity, specificity, fscore

# Carregando a base de dados Breast Cancer Wisconsin
cancer = load_breast_cancer()

# Convertendo os dados e os rótulos para arrays NumPy
X = cancer.data
y = cancer.target

# Listas para armazenar as métricas de cada rodada
cm_list = []
accuracy_list = []
sensitivity_list = []
specificity_list = []
fscore_list = []

# Realizando 10 rodadas de treinamento e teste
for i in range(10):
    # Dividindo os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    
    # Padronizando os dados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Criando e treinando o classificador quadrático
    clf = QuadraticDiscriminantAnalysis()
    clf.fit(X_train_scaled, y_train)
    
    # Fazendo previsões
    y_pred = clf.predict(X_test_scaled)
    
    # Calculando métricas
    cm, accuracy, sensitivity, specificity, fscore = calculate_metrics_and_boxplots(y_test, y_pred)
    cm_list.append(cm)
    accuracy_list.append(accuracy)
    sensitivity_list.append(sensitivity)
    specificity_list.append(specificity)
    fscore_list.append(fscore)
    
    # Imprimindo as métricas
    print(f"Iteração {i+1}:")
    print("Matriz de Confusão:")
    print(cm)
    print("Acurácia:", accuracy)
    print("Sensibilidade (Recall):", sensitivity)
    print("Especificidade:", specificity)
    print("F-score:", fscore)
    print()

# Plotando boxplots das métricas
plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
plt.boxplot(accuracy_list)
plt.title('Accuracy')

plt.subplot(2, 2, 2)
plt.boxplot(sensitivity_list)
plt.title('Sensitivity')

plt.subplot(2, 2, 3)
plt.boxplot(specificity_list)
plt.title('Specificity')

plt.subplot(2, 2, 4)
plt.boxplot(fscore_list)
plt.title('F-score')

plt.tight_layout()
plt.show()
