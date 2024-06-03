import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications, Model # type: ignore
from tensorflow.keras.datasets import cifar10 # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Carregar o conjunto de dados CIFAR-10
(X_treino, y_treino), (X_teste, y_teste) = cifar10.load_data()

# Selecionar apenas as classes 0 (avião) e 1 (automóvel)
classes = [0, 1]
X_treino = X_treino[np.isin(y_treino, classes).flatten()]
y_treino = y_treino[np.isin(y_treino, classes).flatten()]
X_teste = X_teste[np.isin(y_teste, classes).flatten()]
y_teste = y_teste[np.isin(y_teste, classes).flatten()]

# Converter imagens para tons de cinza
X_treino_cinza = np.expand_dims(np.dot(X_treino[...,:3], [0.2989, 0.5870, 0.1140]), axis=-1)
X_teste_cinza = np.expand_dims(np.dot(X_teste[...,:3], [0.2989, 0.5870, 0.1140]), axis=-1)

# Normalizar os valores de pixel para o intervalo [0, 1]
X_treino_cinza = X_treino_cinza.astype('float32') / 255.0
X_teste_cinza = X_teste_cinza.astype('float32') / 255.0

# Dividir o conjunto de treinamento em treinamento e validação
X_treino_cinza, X_val_cinza, y_treino, y_val = train_test_split(X_treino_cinza, y_treino, test_size=0.3, random_state=42)

# Redimensionar as imagens para 32x32
def redimensionar_imagens(imagens, tamanho=(32, 32)):
    imagens_redimensionadas = np.zeros((imagens.shape[0], tamanho[0], tamanho[1], 1))
    for i in range(imagens.shape[0]):
        imagens_redimensionadas[i] = tf.image.resize(imagens[i], tamanho)
    return imagens_redimensionadas

X_treino_cinza = redimensionar_imagens(X_treino_cinza)
X_val_cinza = redimensionar_imagens(X_val_cinza)
X_teste_cinza = redimensionar_imagens(X_teste_cinza)

def treinar_e_avaliar():
    # Carregar o modelo pré-treinado EfficientNetB0
    modelo_base = applications.EfficientNetB0(input_shape=(32, 32, 1), include_top=False, weights=None)
    modelo_base.trainable = True

    # Adicionar camadas finais para ajuste fino
    x = modelo_base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)
    previsoes = layers.Dense(1, activation='sigmoid')(x)

    modelo = Model(inputs=modelo_base.input, outputs=previsoes)

    # Compilar o modelo
    modelo.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Treinar o modelo
    inicio = time.time()
    print("Começou o treino")
    historico = modelo.fit(X_treino_cinza, y_treino, epochs=10, batch_size=64, validation_data=(X_val_cinza, y_val), verbose=0)
    fim = time.time()
    tempo = fim - inicio
    print(tempo)
    tempo_rodada.append(tempo)

    # Avaliar o modelo com o conjunto de teste
    y_pred = modelo.predict(X_teste_cinza)
    y_pred_binario = (y_pred > 0.5).astype('int32')

    # Calcular a matriz de confusão
    matriz_confusao = confusion_matrix(y_teste, y_pred_binario)

    # Calcular métricas
    acuracia = accuracy_score(y_teste, y_pred_binario)
    especificidade = precision_score(y_teste, y_pred_binario)
    sensibilidade = recall_score(y_teste, y_pred_binario)
    f_score = f1_score(y_teste, y_pred_binario)

    return {
        'matriz_confusao': matriz_confusao,
        'acuracia': acuracia,
        'especificidade': especificidade,
        'sensibilidade': sensibilidade,
        'f_score': f_score
    }

resultados = []
tempo_rodada = []
for _ in range(10):
    resultado = treinar_e_avaliar()
    resultados.append(resultado)

tempo_medio = np.mean(tempo_rodada)
print(f"Tempo médio de treino: {tempo_medio:.5f}")
print(f"Maior tempo de um treinamento: {np.max(tempo_rodada):.5f}")
print(f"Menor tempo de um treinamento: {np.min(tempo_rodada):.5f}")

# Identificar as melhores e piores rodadas com base na acurácia
melhor_resultado = max(resultados, key=lambda x: x['acuracia'])
pior_resultado = min(resultados, key=lambda x: x['acuracia'])

# Plotar as matrizes de confusão das melhores e piores rodadas
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(melhor_resultado['matriz_confusao'], cmap='coolwarm', annot=True, linewidth=1, fmt='d', ax=ax[0])
ax[0].set_title(f"Melhor Matriz de Confusão (Acurácia: {melhor_resultado['acuracia']:.2f})")
ax[0].set_xlabel('Predito')
ax[0].set_ylabel('Real')

sns.heatmap(pior_resultado['matriz_confusao'], cmap='coolwarm', annot=True, linewidth=1, fmt='d', ax=ax[1])
ax[1].set_title(f"Pior Matriz de Confusão (Acurácia: {pior_resultado['acuracia']:.2f})")
ax[1].set_xlabel('Predito')
ax[1].set_ylabel('Real')

plt.show()

# Plotar boxplot das métricas
nomes_metricas = ['acuracia', 'especificidade', 'sensibilidade', 'f_score']
valores_metricas = {nome: [resultado[nome] for resultado in resultados] for nome in nomes_metricas}

plt.figure(figsize=(10, 5))
plt.boxplot(valores_metricas.values(), labels=valores_metricas.keys())
plt.title('Boxplot das Métricas')
plt.ylabel('Valor')
plt.show()

# Exibir as métricas das melhores e piores rodadas
print("Melhor rodada")
print(f"Acurácia: {melhor_resultado['acuracia']:.5f}")
print(f"Sensibilidade: {melhor_resultado['sensibilidade']:.5f}")
print(f"Especificidade: {melhor_resultado['especificidade']:.5f}")
print(f"F1-score: {melhor_resultado['f1']:.5f}")

print("\nPior rodada")
print(f"Acurácia: {pior_resultado['acuracia']:.5f}")
print(f"Sensibilidade: {pior_resultado['sensibilidade']:.5f}")
print(f"Especificidade: {pior_resultado['especificidade']:.5f}")
print(f"F1-score: {pior_resultado['f1']:.5f}")
