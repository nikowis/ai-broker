import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import neural_networks.feedforward_model as ffm
import alpha
from sklearn import model_selection

TICKER = 'GOOGL'

api = alpha.AlphaVantage()
df = api.daily_adjusted(TICKER)

base_path = './../../target/neural_networks'

if not os.path.exists(base_path):
    os.makedirs(base_path)

data = df[[alpha.ADJUSTED_CLOSE_COL]]
df = df[[alpha.ADJUSTED_CLOSE_COL]]

# Dimensions of dataset
n = data.shape[0]
p = data.shape[1]

# Make data a np.array

forecast_days = 1
df.dropna(inplace=True)
df.fillna(value=-99999, inplace=True)
df[alpha.LABEL_COL] = df[alpha.ADJUSTED_CLOSE_COL].shift(-forecast_days)

X = np.array(df.drop([alpha.LABEL_COL], 1))
# X = preprocessing.scale(X)
X_lately = X[-forecast_days:]
X = X[:-forecast_days]
df_removed = df.dropna()

y = np.array(df_removed[alpha.LABEL_COL])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# data = data.values
# # Training and test data
# train_start = 0
# train_end = int(np.floor(0.8 * n))
# test_start = train_end + 1
# test_end = n
# data_train = data[np.arange(train_start, train_end), :]
# data_test = data[np.arange(test_start, test_end), :]

# # Scale data
# scaler = MinMaxScaler(feature_range=(-1, 1))
# scaler.fit(data_train)
# data_train = scaler.transform(data_train)
# data_test = scaler.transform(data_test)
#
# # Build X and y
# X_train = data_train[:, 1:]
# y_train = data_train[:, 0]
# X_test = data_test[:, 1:]
# y_test = data_test[:, 0]

model, opt, mse, out, X_plchldr, Y_plchldr = ffm.construct_model(X_train)

# Setup plot
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(111)
line1, = ax1.plot(y_test)
line2, = ax1.plot(y_test * 0.5)
plt.show()

# Fit neural model
batch_size = 256
mse_train = []
mse_test = []

# Run
epochs = 10
for e in range(epochs):

    # Shuffle training data
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    X_train = X_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    # Minibatch training
    for i in range(0, len(y_train) // batch_size):
        start = i * batch_size
        batch_x = X_train[start:start + batch_size]
        batch_y = y_train[start:start + batch_size]
        # Run optimizer with batch
        model.run(opt, feed_dict={X_plchldr: batch_x, Y_plchldr: batch_y})

        # Show progress
        if np.mod(i, 50) == 0:
            # MSE train and test
            mse_train.append(model.run(mse, feed_dict={X_plchldr: X_train, Y_plchldr: y_train}))
            mse_test.append(model.run(mse, feed_dict={X_plchldr: X_test, Y_plchldr: y_test}))
            print('MSE Train: ', mse_train[-1])
            print('MSE Test: ', mse_test[-1])
            # Prediction
            pred = model.run(out, feed_dict={X_plchldr: X_test})
            line2.set_ydata(pred)
            plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
            plt.pause(0.01)
