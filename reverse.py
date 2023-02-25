import pandas as pd

df = pd.read_csv("xx.csv", header = 0)
df2 = df[::-1].set_index(df.index)
df2.to_csv("reversed.csv", sep=',', encoding='utf-8')