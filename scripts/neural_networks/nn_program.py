import os

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Activation
from keras.models import Sequential

import alpha
import neural_networks.data_helper as data_helper

TICKER = 'GOOGL'

api = alpha.AlphaVantage()
df = api.data(TICKER)

base_path = './../../target/neural_networks'

if not os.path.exists(base_path):
    os.makedirs(base_path)

forecast_days = 1
df.dropna(inplace=True)
df.fillna(value=-99999, inplace=True)
df = df[[alpha.ADJUSTED_CLOSE_COL]]
df[alpha.LABEL_COL] = df[alpha.ADJUSTED_CLOSE_COL].shift(-forecast_days)

X = np.array(df.drop([alpha.LABEL_COL], 1))
input_size = X.shape[1]
X_lately = X[-forecast_days:]
X = X[:-forecast_days]
df = df[:-forecast_days]
df_removed = df.dropna()

y = np.array(df_removed[alpha.LABEL_COL])

X_train, X_test, y_train, y_test = data_helper.train_test_split(X, y)

df_cpy = df.copy()

model = Sequential([
    Dense(1, input_shape=(input_size,)),
    Activation('relu'),
    Dense(1),
])

model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=3, batch_size=10)
accuracy = model.evaluate(X_test, y_test)

print("Accuracy: ", accuracy, " epochs: ", 10)

predicted = model.predict(X)

# print('Predicted value ', predicted)

df_cpy[alpha.FORECAST_FUTURE_COL] = predicted

df_plt = df_cpy[[alpha.ADJUSTED_CLOSE_COL, alpha.LABEL_COL, alpha.FORECAST_FUTURE_COL]]
df_plt.plot()
plt.show()
