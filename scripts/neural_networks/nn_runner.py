import os

import numpy as np

import db.stock_constants as const
from db import db_access
from helpers import data_helper
from helpers import plot_helper
from neural_networks import nn_model

base_path = './../../target/neural_networks'

if not os.path.exists(base_path):
    os.makedirs(base_path)

LAYERS = [20, 20, 20]
EPOCHS = 100
FORECAST_DAYS = 1
ACTIVATION = None
OPTIMIZER = 'adam'
LOSS = 'categorical_crossentropy'
BATCH_SIZE = 10
TICKER = 'GOOGL'
HISTORY_DAYS = 20


def run(layers=LAYERS, epochs=EPOCHS, forecast_days=FORECAST_DAYS, activation=ACTIVATION, optimizer=OPTIMIZER,
        loss=LOSS, batch_size=BATCH_SIZE, ticker=TICKER, history_days=HISTORY_DAYS):
    db_conn = db_access.create_db_connection()
    df = db_access.find_one_by_ticker_dateframe(db_conn, ticker)

    df, X_standarized, X_train, X_test, y_train_binary, y_test_binary = data_helper.extract_data(df, forecast_days,
                                                                                                 history_days=history_days)

    model = nn_model.create_seq_model(layers, input_size=X_train.shape[1], activation=activation, optimizer=optimizer,
                                      loss=loss)

    history = model.fit(X_train, y_train_binary, epochs=epochs, batch_size=batch_size)
    loss, accuracy = model.evaluate(X_test, y_test_binary)

    print("Loss: ", loss, " Accuracy: ", accuracy, " epochs: ", epochs)
    predicted_binary = model.predict(X_standarized)
    predicted = [np.argmax(pred, axis=None, out=None) for pred in predicted_binary]
    df[const.FORECAST_DISCRETE_COL] = predicted
    main_title = "Loss: " + str(round(loss, 4)) + ", accuracy: " + str(round(accuracy, 4)) + ", epochs: " + str(
        epochs) + ', history days:' + str(history_days) + '\n'
    main_title += 'Layers: [' + ''.join(
        str(e) + " " for e in layers) + '], optimizer: ' + optimizer + ', loss: ' + str(loss) + ', activation: ' + str(
        activation)
    plot_helper.plot_result(ticker, df, history, main_title)


if __name__ == "__main__":
    run()
