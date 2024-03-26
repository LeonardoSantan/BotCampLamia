import pandas as pd
import numpy as np

df = pd.DataFrame(
    {
        "col1": [1, 2, 3, 4],
        "col2": [444, 555, 666, 444],
        "col3": ["abc", "def", "ghi", "xyz"],
    }
)


# df["col1"]
# df['col1'].unique()


# df['col1'].nunique()
# len(df['col1'].unique())
# df['col2']
# df['col2'].value_counts


# df[(df["col1"] > 2) & (df["col2"] == 444)]


# def vezes2(x):
# return x * 2


# df["col3"].apply(len)

# df["col1"].apply(vezes2)

# df["col1"].apply(lambda x: x * x)


# del df["col2"]

# df

# df.columns

# df.index

# df.sort_values(by='col2')

# df.sort_index()

# df.isnull()

# df.dropna()