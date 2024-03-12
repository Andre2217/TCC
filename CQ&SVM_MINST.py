# Importando as bibliotecas necessárias
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense

# Carregando o conjunto de dados MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalizando os dados
X_train = X_train.reshape(-1, 28*28) / 255.0
X_test = X_test.reshape(-1, 28*28) / 255.0

# Dividindo os dados em conjuntos de treinamento e validação
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Definindo função para construir e treinar os modelos
def build_and_train_model(X_train, y_train, X_val, y_val, model_type):
    # Criando o modelo
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(10, activation='softmax')  # Camada de saída com 10 neurônios para as 10 classes do MNIST
    ])

    # Compilando o modelo
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Treinando o modelo
    model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_val, y_val), verbose=1)

    # Avaliando o modelo
    loss, accuracy = model.evaluate(X_test, y_test)
    print("Acurácia do {} com Keras-TensorFlow: {:.2f}%".format(model_type, accuracy * 100))
    return accuracy

# Treinando e avaliando o modelo SVM
svm_accuracy = build_and_train_model(X_train, y_train, X_val, y_val, "SVM")

# Treinando e avaliando o modelo CQ
cq_accuracy = build_and_train_model(X_train, y_train, X_val, y_val, "CQ")

print("\nAcurácia do SVM: {:.2f}%".format(svm_accuracy * 100))
print("Acurácia do CQ: {:.2f}%".format(cq_accuracy * 100))
