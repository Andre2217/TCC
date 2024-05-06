import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.datasets import mnist # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Carregar o conjunto de dados MNIST e pré-processá-lo
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Filtrar apenas os dígitos 0 e 1
X_train = X_train[(y_train == 0) | (y_train == 1)]
y_train = y_train[(y_train == 0) | (y_train == 1)]
X_test = X_test[(y_test == 0) | (y_test == 1)]
y_test = y_test[(y_test == 0) | (y_test == 1)]

# Normalizar os valores de pixel para o intervalo [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Adicionar uma dimensão de canal para os dados
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Dividir o conjunto de treinamento em treinamento e validação
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Definir o modelo CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compilar o modelo
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Treinar o modelo
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_val, y_val))

# Avaliar o modelo com o conjunto de teste
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype('int32')

# Calcular a matriz de confusão
cm = confusion_matrix(y_test, y_pred_binary)

# Calcular métricas
accuracy = accuracy_score(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)

print("Matriz de Confusão:")
print(cm)
print("Acurácia:", accuracy)
print("Precisão:", precision)
print("Sensibilidade (Recall):", recall)
print("F-score:", f1)

# Plotar boxplot das métricas
metrics = {
    'Accuracy': accuracy_score,
    'Precision': precision_score,
    'Recall': recall_score,
    'F1-Score': f1_score
}

results = {metric_name: [metric(y_test, y_pred_binary)] for metric_name, metric in metrics.items()}

plt.figure(figsize=(10, 5))
plt.boxplot(results.values(), labels=results.keys())
plt.title('Boxplot das Métricas')
plt.ylabel('Valor')
plt.show()
