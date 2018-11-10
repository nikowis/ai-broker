import os

import keras
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style

import db.stock_constants as const
from db import db_access
from helpers import data_helper as data_helper
from helpers import plot_helper
from neural_networks import nn_model

base_path = './../../target/neural_networks'

if not os.path.exists(base_path):
    os.makedirs(base_path)


def run():
    TICKER = 'GOOGL'

    db_conn = db_access.create_db_connection()

    df = db_access.find_one_by_ticker_dateframe(db_conn, TICKER)

    forecast_days = 1

    df, X, y, X_lately = data_helper.prepare_label_extract_data(df, forecast_days)
    X_train, X_test, y_train, y_test = data_helper.train_test_split(X, y, scale=True)

    layers = [5, 5]

    model = nn_model.create_seq_model(layers)

    epochs = 40
    y_train_binary = keras.utils.to_categorical(y_train)
    model.fit(X_train, y_train_binary, epochs=epochs, batch_size=10)
    y_test_binary = keras.utils.to_categorical(y_test)
    loss, accuracy = model.evaluate(X_test, y_test_binary)

    print("Loss: ", loss, " Accuracy: ", accuracy, " epochs: ", epochs)

    predicted_binary = model.predict(X)

    predicted = [np.argmax(pred, axis=None, out=None) for pred in predicted_binary]

    df[const.FORECAST_DISCRETE_COL] = predicted
    style.use('ggplot')
    df[const.LABEL_DISCRETE_COL].plot(kind='hist', xticks=[0, 1, 2], label=plot_helper.RATE_CHANGE_LABEL)
    df[const.FORECAST_DISCRETE_COL].plot(kind='hist', xticks=[0, 1, 2], label=plot_helper.RATE_CHANGE_FORECAST_LABEL)
    plt.xticks([0, 1, 2], [plot_helper.FALL_LABEL, plot_helper.IDLE_LABEL, plot_helper.RISE_LABEL])
    plot_helper.legend_labels_save_files(TICKER, 'nn_discrete_score', base_path, plot_helper.VALUE_CHANGE_LABEL,
                                         plot_helper.FORECAST_COUNT_LABEL, 2)


if __name__ == "__main__":
    run()
