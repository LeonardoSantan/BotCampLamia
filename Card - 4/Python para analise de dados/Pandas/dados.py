import numpy as np
import pandas as pd


d = {"A": [1, 2, np.nan], "B": [5, np.nan, np.nan], "C": [1, 2, 3]}

df = pd.DataFrame(d)

# df.fillna(value=69)

# df['A'].fillna(value=df['A'].sum())
# df["A"].fillna(value=df["A"].mean())
# df["A"].fillna(value=df["A"].max())

# ffill = foward fill
df.fillna(method='ffill')
