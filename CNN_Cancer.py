import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
import matplotlib.pyplot as plt

# Carregando a base de dados Breast Cancer Wisconsin
data = load_breast_cancer()
X, y = data.data, data.target

# Normalizando os dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Definindo a função para calcular as métricas
def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)
    specificity = tn / (tn + fp)
    f1 = f1_score(y_true, y_pred)
    return accuracy, sensitivity, specificity, f1

# Definindo o número de folds para a validação cruzada
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Listas para armazenar as métricas
accuracy_list = []
sensitivity_list = []
specificity_list = []
f1_list = []

# Loop sobre os folds da validação cruzada
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Definindo a arquitetura da CNN
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    # Compilando o modelo
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Treinando o modelo
    model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=0)

    # Prevendo as classes dos dados de teste
    y_pred = model.predict(X_test)
    y_pred_classes = (y_pred > 0.5).astype("int32").flatten()

    # Calculando as métricas
    accuracy, sensitivity, specificity, f1 = calculate_metrics(y_test, y_pred_classes)

    # Armazenando as métricas
    accuracy_list.append(accuracy)
    sensitivity_list.append(sensitivity)
    specificity_list.append(specificity)
    f1_list.append(f1)

# Imprimindo as métricas médias
print(f'Acurácia média: {np.mean(accuracy_list)}')
print(f'Sensibilidade média: {np.mean(sensitivity_list)}')
print(f'Especificidade média: {np.mean(specificity_list)}')
print(f'F-score médio: {np.mean(f1_list)}')

# Plotando os boxplots
plt.figure(figsize=(10, 6))
plt.boxplot([accuracy_list, sensitivity_list, specificity_list, f1_list], labels=['Acurácia', 'Sensibilidade', 'Especificidade', 'F-score'])
plt.title('Distribuição das métricas')
plt.ylabel('Valor')
plt.show()
