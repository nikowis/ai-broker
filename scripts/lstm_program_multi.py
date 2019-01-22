import os
import time

import keras
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential

import api_to_db_importer
import data_helper
import plot_helper
import stock_constants as const

VERBOSE = 2

TARGET_DIR = './../target'
CSV_FILES_DIR = TARGET_DIR + '/data'

DAYS_IN_WINDOW = 7
DROPUT_RATE = 0.2
NEURON_COUNT = 10
STOCK_COMPANIES = 10
BINARY_CLASSIFICATION = True
ACTIVATION = 'softmax'
OPTIMIZER = 'Adam'
LOSS_FUN = 'categorical_crossentropy'
EPOCHS = 5
BATCH_SIZE = 10
COLUMNS = [const.VOLUME_COL, const.OPEN_COL, const.ADJUSTED_CLOSE_COL, const.HIGH_COL, const.LOW_COL,
           const.HL_PCT_CHANGE_COL]


def main():
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)

    model_filepath = TARGET_DIR + '/model.days' + str(
        DAYS_IN_WINDOW) + '.neurons' + str(
        NEURON_COUNT) + '.epochs{epoch:02d}-accuracy{val_categorical_accuracy:.3f}.hdf5'
    total_time = time.time()
    callbacks = [
        # EarlyStopping(monitor='val_loss', min_delta=0.005, patience=20, verbose=0, mode='auto'),
        # ModelCheckpoint(model_filepath, monitor='val_categorical_accuracy', verbose=0, save_best_only=True, mode='auto',
        #                 period=5)
    ]

    symbols = api_to_db_importer.SYMBOLS[0:STOCK_COMPANIES]

    df_list = api_to_db_importer.Importer().import_data_from_files(symbols, CSV_FILES_DIR)

    x_train, x_test, y_train_one_hot, y_test_one_hot = data_helper.extract_data_from_list(df_list, 0,
                                                                                          binary_classification=BINARY_CLASSIFICATION)

    x_train_lstm = prepare_lstm_data(DAYS_IN_WINDOW, x_train)
    x_test_lstm = prepare_lstm_data(DAYS_IN_WINDOW, x_test)
    y_train_one_hot = y_train_one_hot[DAYS_IN_WINDOW - 1:]
    y_test_one_hot = y_test_one_hot[DAYS_IN_WINDOW - 1:]

    _, class_count = y_test_one_hot.shape

    model = Sequential()
    model.add(keras.layers.LSTM(NEURON_COUNT, input_shape=(DAYS_IN_WINDOW, x_train_lstm.shape[2])))

    model.add(keras.layers.Dense(class_count, activation=ACTIVATION))

    model.compile(optimizer=OPTIMIZER,
                  loss=LOSS_FUN,
                  metrics=['categorical_accuracy'])

    history = model.fit(x_train_lstm, y_train_one_hot, validation_data=(x_test_lstm, y_test_one_hot), epochs=EPOCHS,
                        verbose=VERBOSE, batch_size=BATCH_SIZE, callbacks=callbacks)
    loss, accuracy = model.evaluate(x_test_lstm, y_test_one_hot, verbose=VERBOSE)

    history_epochs = len(history.epoch)
    print("Days:", DAYS_IN_WINDOW, " time:", str(int(time.time() - total_time)), " Loss: ", loss, " Accuracy: ",
          accuracy, " epochs: ", history_epochs)

    main_title = get_report_title(accuracy, ACTIVATION, history_epochs, DAYS_IN_WINDOW, loss, LOSS_FUN, OPTIMIZER)

    y_test_score = model.predict(x_test_lstm)

    report_file_name = get_report_file_name(accuracy, DAYS_IN_WINDOW, history_epochs)

    plot_helper.plot_result(y_test_one_hot, y_test_score, class_count, history, main_title,
                            report_file_name, target_dir=TARGET_DIR)

    print("finished " + str(DAYS_IN_WINDOW))


def prepare_lstm_data(DAYS_IN_WINDOW, data):
    lstm_data = np.zeros((data.shape[0] - DAYS_IN_WINDOW + 1, DAYS_IN_WINDOW, data.shape[1]))
    for i in range(DAYS_IN_WINDOW):
        if i == 0:
            train = data[DAYS_IN_WINDOW - 1:]
        else:
            train = data[DAYS_IN_WINDOW - i - 1:-i]
        lstm_data[:, i, :] = train
    return lstm_data


def get_report_title(accuracy, actv, history_epochs, DAYS_IN_WINDOW, loss, lss, optmzr):
    main_title = "Loss: " + str(round(loss, 4)) + ", accuracy: " + str(
        round(accuracy, 4)) + ", epochs: " + str(
        history_epochs) + ', history days:' + str(DAYS_IN_WINDOW) + '\n'
    main_title += 'LSTM: [' + str(NEURON_COUNT) + '], optimizer: ' + str(optmzr) + ', loss: ' + str(
        lss) + ', activation: ' + str(actv)
    return main_title


def get_report_file_name(accuracy, DAYS_IN_WINDOW, history_epochs):
    return str(NEURON_COUNT) + '_HIST_' + str(DAYS_IN_WINDOW) + '_ACCURACY_' + "{0:.3f}".format(
        accuracy) + "_EPOCHS_" + str(history_epochs)


if __name__ == '__main__':
    main()
