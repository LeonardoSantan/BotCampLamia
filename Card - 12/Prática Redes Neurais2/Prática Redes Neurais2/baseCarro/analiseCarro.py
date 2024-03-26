import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

base = pd.read_csv("autos.csv", encoding="ISO-8859-1")
# Exclusão das colunas que não serão utilizadas
base = base.drop("dateCrawled", axis=1)
base = base.drop("dateCreated", axis=1)
base = base.drop("nrOfPictures", axis=1)
base = base.drop("postalCode", axis=1)
base = base.drop("lastSeen", axis=1)

base["name"].value_counts()
# Nome detém de diversos nomes diferentes
# Dos quais não tem relação e apoio nos valores
base = base.drop("name", axis=1)

# Informação não é válida para verifição
# pela quantidade de atributos
base["seller"].value_counts()
base = base.drop("seller", axis=1)

# Valores de oferta referentes a leilão não
# devem ser analisados juntos
base["offerType"].value_counts()
base = base.drop("offerType", axis=1)

# Primeiro passo é sempre a redução de ruido
# afim de tornar nosso algoritimo mais eficiente


# Não faz sentido os valores abaixo de 10 euros para carro
i1 = base.loc[base.price <= 10]

# Validação valor médio de venda
base.price.mean()

# "Exclusão" dos valores do dataframe
base = base[base.price > 10]

# valor acima de 350000
i2 = base.loc[base.price > 350000]

base = base.loc[base.price < 350000]

# Ajuste dos valores NaN
base.loc[pd.isnull(base["vehicleType"])]
base["vehicleType"].value_counts()  # limousine

base.loc[pd.isnull(base["gearbox"])]
base["gearbox"].value_counts()  # manuell

base.loc[pd.isnull(base["model"])]
base["model"].value_counts()  # golf

base.loc[pd.isnull(base["fuelType"])]
base["fuelType"].value_counts()  # benzin

base.loc[pd.isnull(base["notRepairedDamage"])]
base["notRepairedDamage"].value_counts()  # nein

valores = {
    "vehicleType": "limousine",
    "gearbox": "manuell",
    "model": "golf",
    "fuelType": "benzin",
    "notRepairedDamage": "nein",
}

base = base.fillna(value=valores)


previsores = base.iloc[:, 1:13].values
precoReal = base.iloc[:, 0].values

# Transformação de campos categóricos

labelenconderPrevisores = LabelEncoder()

previsores[:, 0] = labelenconderPrevisores.fit_transform(previsores[:, 0])
previsores[:, 1] = labelenconderPrevisores.fit_transform(previsores[:, 1])
previsores[:, 3] = labelenconderPrevisores.fit_transform(previsores[:, 3])
previsores[:, 5] = labelenconderPrevisores.fit_transform(previsores[:, 5])
previsores[:, 8] = labelenconderPrevisores.fit_transform(previsores[:, 8])
previsores[:, 9] = labelenconderPrevisores.fit_transform(previsores[:, 9])
previsores[:, 10] = labelenconderPrevisores.fit_transform(previsores[:, 10])

# Criação da classe Dummy

# 0 -> 0 0 0
# 2 -> 0 1 0
# 3 -> 0 0 1

# Identificação do tipo especifico, afim de categorizar eles

onehotencoder = ColumnTransformer(
    transformers=[("OneHot", OneHotEncoder(), [0, 1, 3, 5, 8, 9, 10])],
    remainder="passthrough",
)
previsores = onehotencoder.fit_transform(previsores).toarray()

regressor = Sequential()
regressor.add(Dense(units=158, activation="relu", input_dim=316))
regressor.add(Dense(units=158, activation="relu"))
# Sigmoid 0 -> 1
# Mais um retorno Softmax
# linear não fará nada
# como queremos somente valor somente ela retornará isto
regressor.add(Dense(units=1, activation="linear"))

# 2 carros base de dados
# primeira previsão 1000(valor real) - 900(valor previsto)
# Segunda previsão 2000(valor real) - 2100(valor previsto)
# Erro total = 100 + 100 (independente se for negativo soma)
# Por conta disso utilizamos o mean_absolute_error
regressor.compile(loss = 'mean_absolute_error', optimizer = 'adam',
                  metrics = ['mean_absolute_error'])

regressor.fit(previsores, precoReal, batch_size = 300, epochs = 100)


previsoes = regressor.predict(previsores)


precoReal.mean()
previsoes.mean()