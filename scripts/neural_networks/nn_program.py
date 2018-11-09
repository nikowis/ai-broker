import os

import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential

from helpers import data_helper as data_helper
from data_processing import alpha

TICKER = 'GOOGL'

api = alpha.AlphaVantage()
df = api.data(TICKER)
df = df[[alpha.ADJUSTED_CLOSE_COL]]

base_path = './../../target/neural_networks'

if not os.path.exists(base_path):
    os.makedirs(base_path)

forecast_days = 1

df, X, y, X_lately = data_helper.prepare_label_extract_data(df, forecast_days)
X_train, X_test, y_train, y_test = data_helper.train_test_split(X, y)

df_cpy = df.copy()

input_size = X.shape[1]
model = Sequential([
    Dense(1, input_shape=(input_size,)),
    Dense(1, )
])

model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])

epochs = 10
model.fit(X_train, y_train, epochs=epochs, batch_size=10)
loss, accuracy = model.evaluate(X_test, y_test)

print("Loss: ", loss, " Accuracy: ", accuracy, " epochs: ", epochs)

predicted = model.predict(X)
df_cpy[alpha.FORECAST_FUTURE_COL] = predicted
df_plt = df_cpy[[alpha.ADJUSTED_CLOSE_COL, alpha.LABEL_COL, alpha.FORECAST_FUTURE_COL]]
df_plt.plot()
plt.show()
