import numpy as np
from keras.models import Sequential
import keras

import data_helper
import db_access
import plot_helper
import stock_constants as const


def main(days_in_window, units):
    ticker = 'USLM'
    db_conn = db_access.create_db_connection(remote=False)
    df_list, sym_list = db_access.find_by_tickers_to_dateframe_parse_to_df_list(db_conn, [ticker])
    df = df_list[0]
    epochs = 200
    batch_size = 1
    skip_iterations = 0

    COLUMNS = [const.VOLUME_COL, const.OPEN_COL, const.ADJUSTED_CLOSE_COL, const.HIGH_COL, const.LOW_COL,
               const.HL_PCT_CHANGE_COL, const.SMA_5_COL, const.SMA_10_COL, const.SMA_20_COL, const.EMA_12_COL,
               const.EMA_26_COL, const.STOCH_K_COL, const.STOCH_D_COL, const.ROC_10_COL, const.TR_COL, const.MOM_6_COL,
               const.MOM_12_COL, const.WILLR_5_COL, const.WILLR_10_COL, const.APO_6_COL, const.APO_12_COL,
               const.RSI_6_COL, const.RSI_12_COL]

    df_modified, x_standardized, x_train, x_test, y_train_one_hot, y_test_one_hot = data_helper.extract_data(df, 0,
                                                                                                             binary_classification=True, input_columns=COLUMNS)

    x_train_lstm = prepare_lstm_data(days_in_window, x_train)
    x_test_lstm = prepare_lstm_data(days_in_window, x_test)
    y_train_one_hot = y_train_one_hot[days_in_window - 1:]
    y_test_one_hot = y_test_one_hot[days_in_window - 1:]

    _, class_count = y_test_one_hot.shape

    model = Sequential()
    model.add(keras.layers.LSTM(units, input_shape=(days_in_window, x_train_lstm.shape[2])))
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Dense(class_count, activation='softmax'))
    model.compile(optimizer='Adam',
                  loss='binary_crossentropy',
                  metrics=['categorical_accuracy'])

    history = model.fit(x_train_lstm, y_train_one_hot, validation_data=(x_test_lstm, y_test_one_hot), epochs=epochs,
                        verbose=0)
    loss, accuracy = model.evaluate(x_test_lstm, y_test_one_hot, verbose=0)

    print("Loss: ", loss, " Accuracy: ", accuracy, " epochs: ", epochs)

    main_title = "Units: " + str(units) + " loss: " + str(round(loss, 4)) + ", accuracy: " + str(round(accuracy, 4)) + ", epochs: " + str(
        epochs) + ', history days:' + str(days_in_window) + '\n'

    y_test_score = model.predict(x_test_lstm)

    plot_helper.plot_result(y_test_one_hot, y_test_score, class_count, history, main_title, 'lstm-days-' + str(days_in_window) + '-units-' + str(units))

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
    unitz= [2, 3,4,5,6]
    for i in range(1, 5):
        for u in unitz:
            main(i, u)

