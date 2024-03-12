# Importando as bibliotecas necessárias
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Carregando o conjunto de dados (vamos usar o conjunto de dados Iris como exemplo)
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando o classificador SVM
svm_classifier = SVC(kernel='linear')

# Treinando o classificador SVM
svm_classifier.fit(X_train, y_train)

# Fazendo previsões
y_pred = svm_classifier.predict(X_test)

# Avaliando a precisão do modelo
accuracy = accuracy_score(y_test, y_pred)
print("Acurácia do SVM:", accuracy)
