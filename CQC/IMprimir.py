import numpy as np
import matplotlib.pyplot as plt

# Carrega o conjunto de dados
Dataset = np.loadtxt("./EMGDataset.csv", delimiter=',')

# Filtra os dados para obter apenas as classes 4 e 5
DadosClasse1 = Dataset[Dataset[:,2] == 4, :]
DadosClasse2 = Dataset[Dataset[:,2] == 5, :]

# Combina os dados filtrados
dados_filtrados = np.vstack((DadosClasse1, DadosClasse2))

# Separa em X (dados) e Y (classes)
X = dados_filtrados[:, :2]  # As duas primeiras colunas
Y = dados_filtrados[:, 2]   # A terceira coluna

# Cria o gráfico de espalhamento
plt.figure(figsize=(10, 6))

# Plota os dados da classe 4 em azul
plt.scatter(X[Y == 4, 0], X[Y == 4, 1], color='blue', label='Classe 4')

# Plota os dados da classe 5 em vermelho
plt.scatter(X[Y == 5, 0], X[Y == 5, 1], color='red', label='Classe 5')

# Adiciona título e rótulos
plt.title('Gráfico de Espalhamento das Classes 4 e 5')
plt.xlabel('Primeira Coluna de Dados')
plt.ylabel('Segunda Coluna de Dados')

# Adiciona legenda
plt.legend()

# Mostra o gráfico
plt.show()
