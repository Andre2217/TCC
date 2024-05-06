import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Carregar os dados do arquivo CSV
data = pd.read_csv('../EMG.csv', header=None)
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Criar e treinar o modelo MLP para classificação com parada antecipada
model = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=10000, early_stopping=True, n_iter_no_change=20, random_state=42)
model.fit(X_train, y_train)

# Avaliar o modelo
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Calcular métricas de avaliação
train_conf_matrix = confusion_matrix(y_train, train_pred)
test_conf_matrix = confusion_matrix(y_test, test_pred)

train_accuracy = accuracy_score(y_train, train_pred)
test_accuracy = accuracy_score(y_test, test_pred)

train_precision = precision_score(y_train, train_pred, average='weighted')
test_precision = precision_score(y_test, test_pred, average='weighted')

train_recall = recall_score(y_train, train_pred, average='weighted')
test_recall = recall_score(y_test, test_pred, average='weighted')

train_f1 = f1_score(y_train, train_pred, average='weighted')
test_f1 = f1_score(y_test, test_pred, average='weighted')

print("Métricas do conjunto de treinamento:")
print("Matriz de Confusão:")
print(train_conf_matrix)
print("Acurácia:", train_accuracy)
print("Precisão:", train_precision)
print("Recall:", train_recall)
print("F-score:", train_f1)

print("\nMétricas do conjunto de teste:")
print("Matriz de Confusão:")
print(test_conf_matrix)
print("Acurácia:", test_accuracy)
print("Precisão:", test_precision)
print("Recall:", test_recall)
print("F-score:", test_f1)

# Criar boxplot das métricas
metric_names = ['Accuracy', 'Precision', 'Recall', 'F-score']
train_metrics = [train_accuracy, train_precision, train_recall, train_f1]
test_metrics = [test_accuracy, test_precision, test_recall, test_f1]

plt.figure(figsize=(10, 6))
sns.boxplot(data=[train_metrics, test_metrics], width=0.5)
plt.xticks(ticks=[0, 1], labels=['Train', 'Test'])
plt.ylabel('Score')
plt.title('Metrics Comparison')
plt.legend(metric_names)
plt.grid(True)
plt.show()
