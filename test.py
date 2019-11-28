import pandas as pd

df = pd.read_csv('dataset/testing-syn.csv',skipinitialspace=True, skiprows=1, nrows=1582682, low_memory=False, index_col=False )
df.to_csv('dataset/syn-dataset.csv', mode='a', header=False, index=False)

