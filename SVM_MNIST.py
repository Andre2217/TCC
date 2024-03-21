# Importando as bibliotecas necessárias
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Carregando o conjunto de dados MNIST
digits = datasets.load_digits()

# Dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

# Inicializando o classificador SVM
svm_classifier = SVC(kernel='linear', gamma='auto')

# Treinando o classificador SVM
svm_classifier.fit(X_train, y_train)

# Prevendo os rótulos para os dados de teste
y_pred = svm_classifier.predict(X_test)

# Calculando a precisão da classificação
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
