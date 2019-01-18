import time

import keras
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential

import api_to_db_importer
import data_helper
import db_access
import plot_helper

DROPUT_RATE = 0.2
NEURON_COUNT = 10
STOCK_COMPANIES = 20
BINARY_CLASSIFICATION = True
actv = 'softmax'
optmzr = 'Adam'
lss = 'categorical_crossentropy'
epochs = 5
batch_size = 5


def main(days_in_window):
    model_filepath = './../target/model.days' + str(
        days_in_window) + '.neurons' + str(
        NEURON_COUNT) + '.epochs{epoch:02d}-accuracy{val_categorical_accuracy:.3f}.hdf5'
    total_time = time.time()
    callbacks = [
        EarlyStopping(monitor='val_loss', min_delta=0.005, patience=20, verbose=0, mode='auto'),
        ModelCheckpoint(model_filepath, monitor='val_categorical_accuracy', verbose=0, save_best_only=True, mode='auto',
                        period=5)
    ]

    symbols = api_to_db_importer.SYMBOLS[0:STOCK_COMPANIES]

    db_conn = db_access.create_db_connection(remote=False)
    df_list, sym_list = db_access.find_by_tickers_to_dateframe_parse_to_df_list(db_conn, symbols)

    x_train, x_test, y_train_one_hot, y_test_one_hot = data_helper.extract_data_from_list(df_list, 0,
                                                                                          binary_classification=BINARY_CLASSIFICATION)

    x_train_lstm = prepare_lstm_data(days_in_window, x_train)
    x_test_lstm = prepare_lstm_data(days_in_window, x_test)
    y_train_one_hot = y_train_one_hot[days_in_window - 1:]
    y_test_one_hot = y_test_one_hot[days_in_window - 1:]

    _, class_count = y_test_one_hot.shape

    model = Sequential()
    model.add(keras.layers.LSTM(NEURON_COUNT, input_shape=(days_in_window, x_train_lstm.shape[2])))
    model.add(keras.layers.Dropout(DROPUT_RATE))

    model.add(keras.layers.Dense(class_count, activation=actv))

    model.compile(optimizer=optmzr,
                  loss=lss,
                  metrics=['categorical_accuracy'])

    history = model.fit(x_train_lstm, y_train_one_hot, validation_data=(x_test_lstm, y_test_one_hot), epochs=epochs,
                        verbose=0, batch_size=batch_size, callbacks=callbacks)
    loss, accuracy = model.evaluate(x_test_lstm, y_test_one_hot, verbose=0)

    history_epochs = len(history.epoch)
    print("Days:", days_in_window, " time:", str(int(time.time() - total_time)), " Loss: ", loss, " Accuracy: ",
          accuracy, " epochs: ", history_epochs)

    main_title = get_report_title(accuracy, actv, history_epochs, days_in_window, loss, lss, optmzr)

    y_test_score = model.predict(x_test_lstm)

    report_file_name = get_report_file_name(accuracy, days_in_window, history_epochs)

    plot_helper.plot_result(y_test_one_hot, y_test_score, class_count, history, main_title,
                            report_file_name,
                            accuracy >= 0.7)

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


def get_report_title(accuracy, actv, history_epochs, days_in_window, loss, lss, optmzr):
    main_title = "Loss: " + str(round(loss, 4)) + ", accuracy: " + str(
        round(accuracy, 4)) + ", epochs: " + str(
        history_epochs) + ', history days:' + str(days_in_window) + '\n'
    main_title += 'LSTM: [' + str(NEURON_COUNT) + '], optimizer: ' + str(optmzr) + ', loss: ' + str(
        lss) + ', activation: ' + str(actv)
    return main_title


def get_report_file_name(accuracy, days_in_window, history_epochs):
    return str(NEURON_COUNT) + + '_HIST_' + str(days_in_window) + '_ACCURACY_' + "{0:.3f}".format(
        accuracy) + "_EPOCHS_" + str(history_epochs)


if __name__ == '__main__':
    for i in range(1, 40):
        main(i)
