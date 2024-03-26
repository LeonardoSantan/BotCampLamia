import numpy as np
import pandas as pd

labels = ["a", "b", "c"]

minha_lista = [10, 20, 30]
arr = np.array([10, 20, 30])
d = {"a": 10, "b": 20, "c": 30}


series = pd.Series(data=minha_lista, index=labels)


series["b"]


ser1 = pd.Series([1, 2, 3, 4], index=["EUA", "ALEMANHA", "URSS", "JAPÃO"])
ser1

ser2 = pd.Series([1, 2, 3, 4], index=["EUA", "ALEMANHA", "ITALIA", "JAPÃO"])
ser2
ser1 + ser2
