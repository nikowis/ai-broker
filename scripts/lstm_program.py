from keras.layers import Dense, LSTM
from keras.models import Sequential

import data_helper
import db_access
import plot_helper
import numpy as np

def main():
    ticker = 'ACUR'
    db_conn = db_access.create_db_connection(remote=False)
    df_list, sym_list = db_access.find_by_tickers_to_dateframe_parse_to_df_list(db_conn, [ticker])
    df = df_list[0]
    epochs = 100
    batch_size = 1
    skip_iterations = 0
    history_days = 10

    df_modified, x_standardized, x_train, x_test, y_train_one_hot, y_test_one_hot = data_helper.extract_data(df,0,
                                                                                                             binary_classification=True)
    x_train = x_train[7:]
    x_test = x_test[5:]
    y_train_one_hot = y_train_one_hot[7:]
    y_test_one_hot = y_test_one_hot[5:]

    x_train = np.reshape(x_train, (int(x_train.shape[0] / 10), 10, x_train.shape[1]))
    x_test = np.reshape(x_test, (int(x_test.shape[0] / 10), 10, x_test.shape[1]))
    y_train_one_hot = y_train_one_hot[::10]
    y_test_one_hot = y_test_one_hot[::10]
    _, class_count = y_test_one_hot.shape

    model = Sequential()
    model.add(LSTM(50, input_shape=(history_days, x_train.shape[2]), return_sequences=False, go_backwards=True))
    model.add(Dense(class_count, activation='softmax'))
    model.compile(optimizer='Adam',
                  loss='binary_crossentropy',
                  metrics=['categorical_accuracy'])

    history = model.fit(x_train, y_train_one_hot, validation_data=(x_test, y_test_one_hot), epochs=epochs,verbose=0)
    loss, accuracy = model.evaluate(x_test, y_test_one_hot, verbose=0)

    print("Loss: ", loss, " Accuracy: ", accuracy, " epochs: ", epochs)

    main_title = "Loss: " + str(round(loss, 4)) + ", accuracy: " + str(round(accuracy, 4)) + ", epochs: " + str(
        epochs) + ', history days:' + str(history_days) + '\n'

    y_test_score = model.predict(x_test)

    plot_helper.plot_result(y_test_one_hot, y_test_score, class_count, history, main_title, 'lstm-test',
                            accuracy >= 0.99)


if __name__ == '__main__':
    main()
