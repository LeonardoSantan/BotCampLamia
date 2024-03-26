
import pandas as pd
import keras_utils
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
base = pd.read_csv("iris.csv")

# Criação de uma variavel somente para previsores e outra para classe
# Iloc trás determinadas informações do nosso dataframe
# No caso abaixo, desejo todas as linhas com as 4 primeiras colunas(5 com indice)
previsores = base.iloc[:, 0:4].values

# Para esta database, a partir da 4 coluna trás as respostas enquanto das anteriores os tamanhos
classe = base.iloc[:, 4].values
previsores

# Modelo de particionamento de uma database
from sklearn.model_selection import train_test_split
previsoresTreinamento, previsoresTeste, classeTreinamento, classeTeste = (
    train_test_split(previsores, classe_dummy, test_size=0.25)
)

from keras.models import Sequential
from keras.layers import Dense
# Construção da estrutura da rede neural
# Atributo classe contem 3 informações. Para ajuste, é necessário realizar processo
from sklearn.preprocessing import LabelEncoder

# Transforma os 3 atributos em 3 valores diferentes
labelecoder = LabelEncoder()
classe = labelecoder.fit_transform(classe)

# Se retorno no ultimo neuronico cair seguintes parametros
# Conseguimos definir o que de fato serão
# iris setosa 1 0 0
# iris virginica 0 1 0
# iris versicolor 0 0 1

from keras.utils import to_categorical
classe_dummy = to_categorical(classe)

def criar_rede():
    classificador = Sequential()
    classificador.add(Dense(units=4, activation="relu", input_dim=4))
    classificador.add(Dense(units=4, activation="relu"))
    classificador.add(Dense(units=3, activation="softmax"))
    classificador.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"]
    )
    return classificador

classificador = KerasClassifier(build_fn=criar_rede,
                                epochs=1000,
                                batch_size=10)

resultados = cross_val_score(estimator = classificador,
                             X = previsores,
                             y= classe,
                             cv = 10,
                             scoring = 'accuracy')

media = resultados.mean()
desvio = resultados.std()
