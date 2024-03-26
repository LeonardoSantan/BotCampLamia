import pandas as pd
from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

base = pd.read_csv(
    "C:/Users/leofe/Documents/LAMIA/Prática Redes Neurais2/baseGames/games.csv"
)

# Identificação de campos que não serão uteis

base = base.drop("Other_Sales", axis=1)
base = base.drop("Global_Sales", axis=1)

# Existe outro campo com informações do Developer já
base = base.drop("Developer", axis=1)


# Tratamento valores faltantes
# Processo será realizado a partir da exclusão dos valores

base = base.dropna(axis=0)
base = base.loc[base["NA_Sales"] > 1]
base = base.loc[base["EU_Sales"] > 1]
base = base.loc[base["NA_Sales"] > 1]


base["Name"].value_counts
nomeJogos = base.Name
base = base.drop("Name", axis=1)

previsores = base.iloc[:, [0, 1, 2, 3, 7, 8, 9, 10, 11]].values

vendaNa = base.iloc[:, 4].values
vendaEu = base.iloc[:, 5].values
vendaJp = base.iloc[:, 6].values

labelencoder = LabelEncoder()
# Ajuste dos valores string em valores numéricos
previsores[0]
previsores[:, 0] = labelencoder.fit_transform(previsores[:, 0])
previsores[:, 2] = labelencoder.fit_transform(previsores[:, 0])
previsores[:, 3] = labelencoder.fit_transform(previsores[:, 0])
previsores[:, 8] = labelencoder.fit_transform(previsores[:, 0])

# Acrescimo de colunas afim de transformar generos em valores 0 e 1
onehotencoder = ColumnTransformer(
    transformers=[("OneHot", OneHotEncoder(), [0, 2, 3, 8])],
    remainder="passthrough",
)
previsores = onehotencoder.fit_transform(previsores).toarray()

camadaEntrada = Input(shape=(65,))
# Como não vamos utilizar um método sequêncial é necessário indicar ligação
# das camadas após fechamento dos parenteses
camadaOculta1 = Dense(
    units=32,
    activation="sigmoid",
)(camadaEntrada)

camadaOculta2 = Dense(units=32, activation="sigmoid")(camadaOculta1)
camadaSaida1 = Dense(units=1, activation="linear")(camadaOculta2)
camadaSaida2 = Dense(units=1, activation="linear")(camadaOculta2)
camadaSaida3 = Dense(units=1, activation="linear")(camadaOculta2)


regressor = Model(
    inputs=camadaEntrada, outputs=[camadaSaida1, camadaSaida2, camadaSaida3]
)
regressor.compile(optimizer="adam", loss="mse")
regressor.fit(previsores, [vendaNa, vendaEu, vendaJp], epochs=5000,
              batch_size=100)

previsaoNa, previsaoEu, previsaoJp = regressor.predict(previsores)
previsaoNa