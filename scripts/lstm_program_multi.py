import time

import keras
import numpy as np
from keras.models import Sequential
from random import randint

import api_to_db_importer
import data_helper
import db_access
import plot_helper


def main(days_in_window):
    total_time = time.time()

    symbols = api_to_db_importer.SYMBOLS[0:20]

    db_conn = db_access.create_db_connection(remote=False)
    df_list, sym_list = db_access.find_by_tickers_to_dateframe_parse_to_df_list(db_conn, symbols)

    epochs = 200
    batch_size = 10
    skip_iterations = 0

    x_train, x_test, y_train_one_hot, y_test_one_hot = data_helper.extract_data_from_list(df_list, 0,
                                                                                          binary_classification=True)

    x_train_lstm = prepare_lstm_data(days_in_window, x_train)
    x_test_lstm = prepare_lstm_data(days_in_window, x_test)
    y_train_one_hot = y_train_one_hot[days_in_window - 1:]
    y_test_one_hot = y_test_one_hot[days_in_window - 1:]

    _, class_count = y_test_one_hot.shape

    model = Sequential()
    model.add(keras.layers.LSTM(40, input_shape=(days_in_window, x_train_lstm.shape[2])))
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Dense(class_count, activation='softmax'))
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    history = model.fit(x_train_lstm, y_train_one_hot, validation_data=(x_test_lstm, y_test_one_hot), epochs=epochs,
                        verbose=0, batch_size=batch_size)
    loss, accuracy = model.evaluate(x_test_lstm, y_test_one_hot, verbose=0)

    print("Days:", days_in_window, " time:", str(int(time.time() - total_time)), " Loss: ", loss, " Accuracy: ",
          accuracy, " epochs: ", epochs)

    main_title = "Loss: " + str(round(loss, 4)) + ", accuracy: " + str(round(accuracy, 4)) + ", epochs: " + str(
        epochs) + ', history days:' + str(days_in_window) + '\n'

    y_test_score = model.predict(x_test_lstm)

    plot_helper.plot_result(y_test_one_hot, y_test_score, class_count, history, main_title,
                            'lstm-test-7-days-' + str(randint(0, 99999)),
                            accuracy >= 0.99)

    print("finished " + str(days_in_window))


def prepare_lstm_data(days_in_window, data):
    lstm_data = np.zeros((data.shape[0] - days_in_window + 1, days_in_window, data.shape[1]))
    for i in range(days_in_window):
        if i == 0:
            train = data[days_in_window - 1:]
        else:
            train = data[days_in_window - i - 1:-i]
        lstm_data[:, i, :] = train
    return lstm_data


if __name__ == '__main__':
    #for i in range(1, 10):
        # main(i)
    main(7)
