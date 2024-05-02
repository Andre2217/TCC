import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Carregar a base de dados Breast Cancer Wisconsin
breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target

# Listas para armazenar as métricas de cada rodada
cm_list = []
accuracy_list = []
sensitivity_list = []
specificity_list = []
fscore_list = []

num_rounds = 10  # Número de rodadas

for n in range(num_rounds):
    # Dividir novamente os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Criar e treinar o modelo SVM
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train, y_train)

    # Fazer previsões nos dados de teste
    y_pred = svm_model.predict(X_test)

    # Calcular a matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    
    # Calcular as métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    specificity = recall_score(y_test, y_pred, pos_label=0)
    f1 = f1_score(y_test, y_pred)
    
    cm_list.append(cm)
    accuracy_list.append(accuracy)
    sensitivity_list.append(recall)
    specificity_list.append(specificity)
    fscore_list.append(f1)
    
    # Imprimir as métricas
    print(f"Iteração {n+1}:")
    print("Matriz de confusão:")
    print(cm)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall (Sensibility):", recall)
    print("Specificity:", specificity)
    print("F1 Score:", f1)
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
