import numpy as np
import matplotlib.pyplot as plt

# Carrega o conjunto de dados
Dataset = np.loadtxt("./EMGDataset.csv", delimiter=',')

# Separa em X (dados) e Y (classes)
X = Dataset[:, :2]  # As duas primeiras colunas
Y = Dataset[:, 2]   # A terceira coluna

# Cria o gráfico de espalhamento
plt.figure(figsize=(10, 6))

# Define as cores para cada classe
cores = {1: 'darkgreen', 2: 'purple', 3: 'orange', 4: 'blue', 5: 'red'}

# Plota os dados de cada classe com sua respectiva cor
for classe in range(1, 6):
    plt.scatter(X[Y == classe, 0], X[Y == classe, 1], color=cores[classe], label=f'Classe {classe}')

# Adiciona título e rótulos
plt.title('Gráfico de Espalhamento das Classes')
plt.xlabel('Primeira Coluna de Dados')
plt.ylabel('Segunda Coluna de Dados')

# Adiciona legenda
plt.legend()

# Mostra o gráfico
plt.show()
