import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications, Model #type:ignore
from tensorflow.keras.datasets import mnist #type:ignore
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Carregar o conjunto de dados MNIST e pré-processá-lo
(X_treino, y_treino), (X_teste, y_teste) = mnist.load_data()

# Filtrar apenas os dígitos 0 e 1
X_treino = X_treino[(y_treino == 0) | (y_treino == 1)]
y_treino = y_treino[(y_treino == 0) | (y_treino == 1)]
X_teste = X_teste[(y_teste == 0) | (y_teste == 1)]
y_teste = y_teste[(y_teste == 0) | (y_teste == 1)]

# Normalizar os valores de pixel para o intervalo [0, 1]
X_treino = X_treino.astype('float32') / 255.0
X_teste = X_teste.astype('float32') / 255.0

# Adicionar uma dimensão de canal para os dados
X_treino = np.expand_dims(X_treino, axis=-1)
X_teste = np.expand_dims(X_teste, axis=-1)

# Dividir o conjunto de treinamento em treinamento e validação
X_treino, X_validacao, y_treino, y_validacao = train_test_split(X_treino, y_treino, test_size=0.1, random_state=42)

# Redimensionar as imagens para 32x32
def redimensionar_imagens(imagens, tamanho=(32, 32)):
    imagens_redimensionadas = np.zeros((imagens.shape[0], tamanho[0], tamanho[1], 1))
    for i in range(imagens.shape[0]):
        imagens_redimensionadas[i] = tf.image.resize(imagens[i], tamanho)
    return imagens_redimensionadas

X_treino = redimensionar_imagens(X_treino)
X_validacao = redimensionar_imagens(X_validacao)
X_teste = redimensionar_imagens(X_teste)

def treinar_e_avaliar():
    # Carregar o modelo pré-treinado EfficientNetB0
    modelo_base = applications.EfficientNetB0(input_shape=(32, 32, 1), include_top=False, weights=None)
    modelo_base.trainable = True

    # Adicionar camadas finais para ajuste fino
    x = modelo_base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(10, activation='relu')(x)
    previsoes = layers.Dense(1, activation='sigmoid')(x)

    modelo = Model(inputs=modelo_base.input, outputs=previsoes)

    # Compilar o modelo
    modelo.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

    # Treinar o modelo
    inicio = time.time()
    print("Começou o treino")
    historico = modelo.fit(X_treino, y_treino, epochs=5, batch_size=64, validation_data=(X_validacao, y_validacao), verbose=0)
    fim = time.time()
    tempo = fim - inicio
    print(tempo)
    tempo_rodada.append(tempo)
    

    # Avaliar o modelo com o conjunto de teste
    y_previsao = modelo.predict(X_teste)
    y_previsao_binaria = (y_previsao > 0.5).astype('int32')

    # Calcular a matriz de confusão
    matriz_confusao = confusion_matrix(y_teste, y_previsao_binaria)

    # Calcular métricas
    acuracia = accuracy_score(y_teste, y_previsao_binaria)
    especificidade = precision_score(y_teste, y_previsao_binaria)
    sensibilidade = recall_score(y_teste, y_previsao_binaria)
    f1 = f1_score(y_teste, y_previsao_binaria)

    return {
        'matriz_confusao': matriz_confusao,
        'acuracia': acuracia,
        'especificidade': especificidade,
        'sensibilidade': sensibilidade,
        'f1': f1
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
ax[0].set_title(f"Melhor Matriz de Confusão (Acurácia: {melhor_resultado['acuracia']:.5f})")
ax[0].set_xlabel('Predito')
ax[0].set_ylabel('Real')

sns.heatmap(pior_resultado['matriz_confusao'], cmap='coolwarm', annot=True, linewidth=1, fmt='d', ax=ax[1])
ax[1].set_title(f"Pior Matriz de Confusão (Acurácia: {pior_resultado['acuracia']:.5f})")
ax[1].set_xlabel('Predito')
ax[1].set_ylabel('Real')

plt.show()

# Plotar boxplot das métricas
nomes_metricas = ['acuracia', 'especificidade', 'sensibilidade', 'f1']
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
