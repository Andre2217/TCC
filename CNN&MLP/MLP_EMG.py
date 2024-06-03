import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential #type:ignore
from tensorflow.keras.layers import Dense, Input #type:ignore
from tensorflow.keras.optimizers import Adam #type:ignore
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Carregar os dados
dataset = np.loadtxt("./EMGDataset.csv", delimiter=',')
dados_classe1 = dataset[dataset[:,2]==4,:]
dados_classe2 = dataset[dataset[:,2]==5,:]
    
dados_filtrados = np.vstack((dados_classe1, dados_classe2))

# Separa em X e Y
X = dados_filtrados[:, :2]  # As duas primeiras colunas
Y = dados_filtrados[:, 2]   # A terceira coluna
    
# Transformar Y em uma matriz binária (one-hot encoding)
Y_binario = np.zeros((Y.size, 2))
Y_binario[Y == 4, 0] = 1
Y_binario[Y == 5, 1] = 1

N, p = X.shape

# Definindo as listas para armazenar as métricas
acuracias = []
sensibilidades = []
especificidades = []
f1_scores = []
matrizes_confusao = []
tempo_rodada=[]

# Loop para realizar 10 rodadas de treinamento e avaliação
for rodada in range(10):
    print(f"Rodada {rodada + 1}")
    
    # Dividir os dados em conjuntos de treinamento e teste
    seed = np.random.permutation(N)
    Xr = np.copy(X[seed, :])
    Yr = np.copy(Y_binario[seed])

    X_treino = Xr[0:int(N * .8), :]
    Y_treino = Yr[0:int(N * .8), :]

    X_teste = Xr[int(N * .8):, :]
    Y_teste = Yr[int(N * .8):, :]

    # Construir o modelo MLP
    model = Sequential()
    model.add(Input(shape=(2,)))  # Definir a entrada com a dimensão correta
    model.add(Dense(10, activation='relu'))  # Camada oculta com 10 neurônios e função de ativação ReLU
    model.add(Dense(10, activation='relu'))  # Segunda camada oculta
    model.add(Dense(2, activation='softmax'))  # Camada de saída com 2 neurônios e função de ativação softmax

    # Compilar o modelo
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

    # Treinar o modelo
    inicio = time.time()
    print(f"Começou o treino da Rodada {rodada+1}")
    history = model.fit(X_treino, Y_treino, epochs=10, batch_size=10, validation_data=(X_teste, Y_teste), verbose=0)
    fim = time.time()
    tempo = fim - inicio
    print(tempo)
    tempo_rodada.append(tempo)

    # Avaliar o modelo
    loss, accuracy = model.evaluate(X_teste, Y_teste, verbose=0)
    
    # Predições no conjunto de teste
    predictions = model.predict(X_teste)
    predictions = np.argmax(predictions, axis=1)
    Y_teste_labels = np.argmax(Y_teste, axis=1)

    # Calcular a matriz de confusão
    cm = confusion_matrix(Y_teste_labels, predictions)
    matrizes_confusao.append(cm)

    # Calcular métricas
    acc = accuracy_score(Y_teste_labels, predictions)
    sens = recall_score(Y_teste_labels, predictions, pos_label=1)
    spec = recall_score(Y_teste_labels, predictions, pos_label=0)
    f1 = f1_score(Y_teste_labels, predictions, pos_label=1)

    acuracias.append(acc)
    sensibilidades.append(sens)
    especificidades.append(spec)
    f1_scores.append(f1)

# Plotar boxplot das métricas
plt.figure(figsize=(10, 7))
plt.boxplot([acuracias, sensibilidades, especificidades, f1_scores], labels=['Acurácia', 'Sensibilidade', 'Especificidade', 'F1-score'])
plt.title('Boxplot das Métricas')
plt.show()

# Identificar a melhor e a pior rodada
melhor_rodada = np.argmax(acuracias)
pior_rodada = np.argmin(acuracias)

# Plotar a melhor matriz de confusão
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(matrizes_confusao[melhor_rodada], annot=True, fmt='d', cmap='coolwarm', ax=ax[0])
ax[0].set_title(f'Melhor Matriz de Confusão (Acurácia: {acuracias[melhor_rodada]:.5f})')
ax[0].set_xlabel('Predito')
ax[0].set_ylabel('Verdadeiro')

# Plotar a pior matriz de confusão
sns.heatmap(matrizes_confusao[pior_rodada], annot=True, fmt='d', cmap='coolwarm', ax=ax[1])
ax[1].set_title(f'Pior Matriz de Confusão (Acurácia: {acuracias[pior_rodada]:.5f})')
ax[1].set_xlabel('Predito')
ax[1].set_ylabel('Verdadeiro')

plt.show()

# Exibir tempo de treino
tempo_medio = np.mean(tempo_rodada)
print(f"Tempo médio de treino: {tempo_medio:.5f}")
print(f"Maior tempo de um treinamento: {np.max(tempo_rodada):.5f}")
print(f"Menor tempo de um treinamento: {np.min(tempo_rodada):.5f}")

# Exibir as métricas das melhores e piores rodadas
print(f"Melhor rodada (Rodada {melhor_rodada + 1}):")
print(f"Acurácia: {acuracias[melhor_rodada]:.5f}")
print(f"Sensibilidade: {sensibilidades[melhor_rodada]:.5f}")
print(f"Especificidade: {especificidades[melhor_rodada]:.5f}")
print(f"F1-score: {f1_scores[melhor_rodada]:.5f}")

print(f"\nPior rodada (Rodada {pior_rodada + 1}):")
print(f"Acurácia: {acuracias[pior_rodada]:.5f}")
print(f"Sensibilidade: {sensibilidades[pior_rodada]:.5f}")
print(f"Especificidade: {especificidades[pior_rodada]:.5f}")
print(f"F1-score: {f1_scores[pior_rodada]:.5f}")
