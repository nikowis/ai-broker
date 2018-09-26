import os

import numpy as np
import tensorflow as tf
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler

import alpha

TICKER = 'GOOGL'

api = alpha.AlphaVantage()
df = api.data(TICKER, api.DataType.INTRADAY, False)

base_path = './../../target/neural_networks'

if not os.path.exists(base_path):
    os.makedirs(base_path)

mnist = tf.keras.datasets.mnist
forecast_days = 1
df.dropna(inplace=True)
df.fillna(value=-99999, inplace=True)
df = df[[alpha.CLOSE_COL, alpha.VOLUME_INTRADAY_COL]]
df[alpha.LABEL_COL] = df[alpha.CLOSE_COL].shift(-forecast_days)

X = np.array(df.drop([alpha.LABEL_COL], 1))
input_size = X.shape[1]
X_lately = X[-forecast_days:]
X = X[:-forecast_days]

df_removed = df.dropna()

y = np.array(df_removed[alpha.LABEL_COL])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
expanded_y_train = np.expand_dims(y_train, axis=1)
expanded_y_test = np.expand_dims(y_test, axis=1)

scaler = MinMaxScaler()

train_data = np.append(X_train, expanded_y_train, axis=1)
test_data = np.append(X_test, expanded_y_test, axis=1)
scaler.fit(train_data)

train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

X_train = train_data[:, :2]
y_train = train_data[:, 2]
X_test = test_data[:, :2]
y_test = test_data[:, 2]

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(8, activation=tf.nn.relu, input_shape=(input_size,)),
    tf.keras.layers.Dense(4, activation=tf.nn.relu),
    tf.keras.layers.Dense(1, activation=tf.nn.relu)
])
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10)
accuracy = model.evaluate(X_test, y_test)

print("Accuracy: ", accuracy)

a = model.predict(X_lately)

print('Predicted value ', a)
