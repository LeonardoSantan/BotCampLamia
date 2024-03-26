import pandas as pd

data = {
    "Empresa": [
        "GOOG",
        "GOOG",
        "MSFT",
        "MSFT",
        "FB",
        "FB",
    ],
    "Nome": ['Sam', 'Charlie', 'Amy', 'Vanessa', 'Carl', 'Sarah'],
    "Venda": [200, 120, 340, 124, 243, 350],
}

df = pd.DataFrame(data)

group = df.groupby('Empresa')

group

group.sum('venda')

group.describe()

group = df.groupby("Nome")

group.sum('Venda')
