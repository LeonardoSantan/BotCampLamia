import numpy as np
import pandas as pd

outside = ["G1", "G1", "G1", "G2", "G2", "G2"]
inside = [1, 2, 3, 1, 2, 3]
hier_index = list(zip(outside, inside))
hier_index = pd.MultiIndex.from_tuples(hier_index)


df = pd.DataFrame(np.random.randn(6, 2), index=hier_index, columns=["A", "B"])

df.loc["G1"].loc[1]

df.index.names = ["Grupo", "Número"]
df

df.xs("G1")

df.xs("G1").xs(1)

df.xs("G1", level="Grupo")
