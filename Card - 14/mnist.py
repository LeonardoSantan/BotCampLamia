import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D

# Etapa 1 - Operador de convolução

# Entrada dos dados para 2 tuplas
(X_treinamento, y_treinamento), (X_teste, y_teste) = mnist.load_data()

# Visualização da imagem
# Atributo cmap define imagem em escala de cinza
plt.imshow(X_treinamento[7], cmap="gray")
plt.title("Classe " + str(y_treinamento[0]))

# Mudança do formato de nossos dados
previsores_treinamento = X_treinamento.reshape(
    X_treinamento.shape[0],
    28,  # Altura
    28,  # Largura
    1,  # Canal
)


previsores_teste = X_teste.reshape(X_teste.shape[0], 28, 28, 1)

# Alterar tipo dos dados
previsores_treinamento = previsores_treinamento.astype("float32")
previsores_teste = previsores_teste.astype("float32")

# Números muito altos do provisores_treinamento
# Iremos diminuir escala de 0-250 para 0-1

previsores_treinamento /= 255
previsores_teste /= 255


classe_treinamento = to_categorical(y_treinamento, 10)
classe_teste = to_categorical(y_teste, 10)

classificador = Sequential()

# Padrão é a utilização de multiplos de 64
# 3.3 é o tamanho do detector de caracteristicas
classificador.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation="relu"))

# Etapa 2 - Polling

classificador.add(MaxPooling2D(pool_size=(2, 2)))

classificador.add(Flatten())

classificador.add(
    Dense(
        units=128,
        activation="relu",
    )
)

classificador.add(Dense(units=10, activation='softmax'))
classificador.compile(loss= 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

classificador.fit(previsores_treinamento, classe_treinamento, batch_size=128, epochs = 5, validation_data = (previsores_teste, classe_teste))


resultado = classificador.evaluate(previsores_teste, classe_teste)