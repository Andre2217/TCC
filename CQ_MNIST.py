import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import MiniBatchKMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Carregar os dados MNIST
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist['data'], mnist['target']

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline para o modelo CQ
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Normalização dos dados
    ('quantizer', MiniBatchKMeans(n_clusters=100)),  # Quantização
    ('classifier', LogisticRegression(max_iter=1000))  # Classificador
])

# Treinar o modelo
pipeline.fit(X_train, y_train)

# Avaliar o modelo
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
