import pandas as pd

# import numpy as np
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, accuracy_score

previsores = pd.read_csv("entradas_breast.csv")
classe = pd.read_csv("saidas_breast.csv")


previsoresTreinamento, previsoresTest, classeTreinamento, classeTeste = (
    train_test_split(previsores, classe, test_size=0.25)
)

len(previsores.columns)
classificador = Sequential()
classificador.add(
    Dense(
        units=16,
        activation="relu",
        kernel_initializer="random_uniform",
        input_dim=30,
    )
)

classificador.add(
    Dense(
        units=16,
        activation="relu",
        kernel_initializer="random_uniform",
    )
)

classificador.add(
    Dense(
        units=1,
        activation="sigmoid",  # retorno de valor 0 e 1
    )
)


otimizador = keras.optimizers.Adam(
    lear=0.001, decay=0.0001, clipvalue=0.5
)

classificador.compile(
    optimizer=otimizador,
    # optimizador que melhor se encaixa nas mais diversas situacoes
    loss="binary_crossentropy",
    metrics=["binary_accuracy"],
)
# encaixar info
classificador.fit(previsoresTreinamento, classeTreinamento, batch_size=10, epochs=100)

pesos0 = classificador.layers[0].get_weights()


previsoes = classificador.predict(previsoresTest)

previsoes = previsoes > 0.5
previsao = accuracy_score(classeTeste, previsoes)
previsao
matriz = confusion_matrix(classeTeste, previsoes)
matriz

resultado = classificador.evaluate(previsoresTest, classeTeste)

resultado
