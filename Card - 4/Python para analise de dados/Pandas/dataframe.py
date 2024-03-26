import numpy as np
import pandas as pd


np.random.seed(101)

df = pd.DataFrame(
    np.random.randn(5, 4), index="A B C D E".split(), columns="W X Y Z".split()
)

# df

# bol = df > 0

# df[bol]

# df[df["W"] > 0]

# df[df["W"] > 0]["Y"]

# df[(df["W"] > 0) & (df["Y"] > 1)]


# df
df.reset_index(inplace=True)

col = "RS RJ SP AM SC".split()

df["Estado"] = col

df.set_index('Estado', inplace=True)

df
