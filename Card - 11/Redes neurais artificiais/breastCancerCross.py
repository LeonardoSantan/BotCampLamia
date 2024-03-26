import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


previsores = pd.read_csv("entradas_breast.csv")
classe = pd.read_csv("saidas_breast.csv")

previsoresTreinamento, previsoresTest, classeTreinamento, classeTeste = (
    train_test_split(previsores, classe, test_size=0.25)
)


def criarRede():
    classificador = Sequential()
    classificador.add(
        Dense(
            units=16,
            activation="relu",
            kernel_initializer="random_uniform",
            input_dim=30,
        )
    )
    classificador.add(Dropout(0.2))
    classificador.add(
        Dense(
            units=16,
            activation="relu",
            kernel_initializer="random_uniform",
        )
    )
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units=1, activation="sigmoid"))
    otimizador = keras.optimizers.Adam(
        learning_rate=0.001, weight_decay=0.0001, clipvalue=0.5
    )
    
    classificador.compile(
        optimizer=otimizador, loss="binary_crossentropy",
        metrics=["binary_accuracy"]
    )
    return classificador


classificador = KerasClassifier(model=criarRede, epochs=100, batch_size=10)

resultados = cross_val_score(
    estimator=classificador,
    X=previsores,
    y=classe,
    cv=10,  # quantas vezes será rodado teste
    scoring="precision",  # Substitua "loss" pela métrica escolhida
)

