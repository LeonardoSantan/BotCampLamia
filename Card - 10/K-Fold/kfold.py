import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn import svm


iris = datasets.load_iris()

# Definiu para variavel train e test 40% dos dados, logo, dividindo eles em 2 grupos
x_train, x_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.4, random_state=0
)

# Construiu um modelo SVC para predição de classificação de iris usando dados de treino
clf = svm.SVC(kernel="linear", C=1).fit(x_train, y_train)

# Agora mediu a performace dos dados testados
clf.score(x_test, y_test)

# Dar para a função cross_val_score, o data set completo, com os valores "reais" e o numero de "dobras"
scores = cross_val_score(clf, iris.data, iris.target, cv=5)

# printar a acuracia de cada dobra
print(scores)

# a média de acuracia das 5 dobras
print(scores.mean())

# Utilizando outro Kernel
clf = svm.SVC(kernel="poly", degree=4, C=1).fit(x_train, y_train)
scores = cross_val_score(clf, iris.data, iris.target, cv=5)

print(scores)
print(scores.mean())


