import os

import keras
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler

import db.stock_constants as const
from db import db_access
from helpers import data_helper as data_helper
from helpers import plot_helper
from neural_networks import nn_model

base_path = './../../target/neural_networks'

if not os.path.exists(base_path):
    os.makedirs(base_path)

LAYERS = [10, 10, 10]
EPOCHS = 50
FORECAST_DAYS = 1
SCALE = True
ACTIVATION = None
OPTIMIZER = 'adam'
LOSS = 'mse'
BATCH_SIZE = 5


def run():
    TICKER = 'GOOGL'

    db_conn = db_access.create_db_connection()

    df = db_access.find_one_by_ticker_dateframe(db_conn, TICKER)

    df, X, y, X_lately = data_helper.prepare_label_extract_data(df, FORECAST_DAYS)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
    X_standarized = X
    if SCALE:
        if len(X_train.shape) == 1:
            X_train = np.expand_dims(X_train, axis=1)
            X_test = np.expand_dims(X_test, axis=1)
        std_scale = StandardScaler().fit(X_train)
        X_train = std_scale.transform(X_train)
        X_test = std_scale.transform(X_test)
        X_standarized = std_scale.transform(X)

    model = nn_model.create_seq_model(LAYERS, input_size=X_train.shape[1], activation=ACTIVATION, optimizer=OPTIMIZER,
                                      loss=LOSS)

    y_train_binary = keras.utils.to_categorical(y_train)
    model.fit(X_train, y_train_binary, epochs=EPOCHS, batch_size=BATCH_SIZE)
    y_test_binary = keras.utils.to_categorical(y_test)
    loss, accuracy = model.evaluate(X_test, y_test_binary)

    print("Loss: ", loss, " Accuracy: ", accuracy, " epochs: ", EPOCHS)

    predicted_binary = model.predict(X_standarized)

    predicted = [np.argmax(pred, axis=None, out=None) for pred in predicted_binary]

    df[const.FORECAST_DISCRETE_COL] = predicted
    style.use('ggplot')
    df[const.LABEL_DISCRETE_COL].plot(kind='hist', xticks=[0, 1, 2], label=plot_helper.RATE_CHANGE_LABEL)
    plot_helper.legend_labels_save_files(TICKER, 'nn_discrete_labels', base_path, plot_helper.VALUE_CHANGE_LABEL,
                                         plot_helper.FORECAST_COUNT_LABEL, 2)
    plt.close()
    df[const.FORECAST_DISCRETE_COL].plot(kind='hist', xticks=[0, 1, 2], label=plot_helper.RATE_CHANGE_FORECAST_LABEL)
    plt.xticks([0, 1, 2], [plot_helper.FALL_LABEL, plot_helper.IDLE_LABEL, plot_helper.RISE_LABEL])
    plot_helper.legend_labels_save_files(TICKER, 'nn_discrete_predictions', base_path, plot_helper.VALUE_CHANGE_LABEL,
                                         plot_helper.FORECAST_COUNT_LABEL, 2)
    plt.close()


if __name__ == "__main__":
    run()
