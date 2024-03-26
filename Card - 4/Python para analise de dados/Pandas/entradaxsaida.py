import numpy as np
import pandas as pd


# df = pd.read_csv('exemplo',sep=',')


# df


# df = df + 1
# df.to_csv('exemplo_csv',sep=';', decimal=',')

# df = pd.read_excel("exemplo_excel.xlsx", sheet_name="Sheet1")

# df


# df.to_excel("exemplo_xlsx.xlsx", sheet_name="Sheet1")

df = pd.read_html("https://www.fdic.gov/resources/resolutions/bank-failures/failed-bank-list/")


df[0]