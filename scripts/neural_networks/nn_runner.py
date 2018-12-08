import numpy as np

import db.stock_constants as const
from db import db_access
from helpers import plot_helper, data_helper
from neural_networks import nn_model

LAYERS = [20, 20, 20]
EPOCHS = 5
FORECAST_DAYS = 1
ACTIVATION = None
OPTIMIZER = 'adam'
LOSS_FUN = 'sparse_categorical_crossentropy'
BATCH_SIZE = 10
TICKER = 'GOOGL'
HISTORY_DAYS = 0


def run(df, x, x_train, x_test, y_train_binary, y_test_binary, layers=LAYERS, epochs=EPOCHS,
        activation=ACTIVATION, optimizer=OPTIMIZER,
        loss_fun=LOSS_FUN, batch_size=BATCH_SIZE, history_days=HISTORY_DAYS, file_name='model_stats',
        outstanding_treshold=0.42):
    _, classes_count = y_test_binary.shape
    model = nn_model.create_seq_model(layers, input_size=x_train.shape[1], activation=activation, optimizer=optimizer,
                                      loss=loss_fun, class_count=classes_count)

    history = model.fit(x_train, y_train_binary, validation_data=(x_test, y_test_binary), epochs=epochs,
                        batch_size=batch_size, verbose=0)
    loss, accuracy = model.evaluate(x_test, y_test_binary, verbose=0)
    print("Loss: ", loss, " Accuracy: ", accuracy, " epochs: ", epochs)
    predicted = model.predict_classes(x)
    df[const.FORECAST_DISCRETE_COL] = predicted
    count_matching = np.count_nonzero(np.where(df[const.FORECAST_DISCRETE_COL] == df[const.LABEL_DISCRETE_COL], 1, 0))
    print('Accuracy on full data:', count_matching / len(predicted))

    main_title = "Loss: " + str(round(loss, 4)) + ", accuracy: " + str(round(accuracy, 4)) + ", epochs: " + str(
        epochs) + ', history days:' + str(history_days) + '\n'
    main_title += 'Layers: [' + ''.join(
        str(e) + " " for e in layers) + '], optimizer: ' + optimizer + ', loss: ' + str(
        loss_fun) + ', activation: ' + str(
        activation)
    y_test_score = model.predict(x_test)

    plot_helper.plot_result(df, y_test_binary, y_test_score, classes_count, history, main_title, file_name,
                            accuracy >= outstanding_treshold)


if __name__ == '__main__':
    db_conn = db_access.create_db_connection()
    df = db_access.find_one_by_ticker_dateframe(db_conn, TICKER)
    df, x_standarized, x_train, x_test, y_train_binary, y_test_binary = data_helper.extract_data(df, 1, HISTORY_DAYS)

    run(df, x_standarized, x_train, x_test, y_train_binary, y_test_binary)
