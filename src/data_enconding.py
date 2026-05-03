import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

COLS = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
df = pd.read_csv("data/car.data", header=None, names=COLS)

# Todos os atributos categóricos em ordem de menos para mais magnitude (por coincidência, todos os atributos desse dataset são categóricos ordinais)
categories = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
orders = [
            ['low', 'med', 'high', 'vhigh'], 
            ['low', 'med', 'high', 'vhigh'], 
            ['2', '3', '4', '5more'], 
            ['2', '4', 'more'], 
            ['small', 'med', 'big'], 
            ['low', 'med', 'high']
        ]

# encoder associa cada valor de cada ordem a um número de 0 a n-1 e então atualizamos o dataframe
encoder = OrdinalEncoder(categories=orders, dtype=np.int64)
encoder.set_output(transform="pandas") # sem isso encoder retorna um numpy array e não um dataframe pandas
df = encoder.fit_transform(df[categories])

df.to_csv('data/car_ord_coded.data', index=False)