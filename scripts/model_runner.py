import time
import uuid

from sklearn import model_selection
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import LabelEncoder

import data_preprocessing
import db_access
import nn_model
import plot_helper

MIN_DATE = '2011-01-01'
MAX_DATE = '2020-10-29'
SELECTED_SYM = 'GOOGL'


def prepare_data(pca=0.999):
    iter_time = time.time()
    global df, x_train, x_test, y_train, y_test
    df, x, y = data_preprocessing.preprocess(df, pca_variance_ratio=pca)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, shuffle=False)
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_Y = encoder.transform(y)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, encoded_Y, test_size=0.2, shuffle=False)
    print('Data processing time ', str(int(time.time() - iter_time)), 's.')
    return x_train, x_test, y_train, y_test


def run(model, x_train, x_test, y_train, y_test, epochs=5, batch_size=5, file_name="test_model_" + uuid.uuid4().hex,
        additional_title_info=None):
    iter_time = time.time()

    history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                        epochs=epochs, batch_size=batch_size, verbose=0)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print("Loss: ", loss, " Accuracy: ", accuracy, " epochs: ", epochs)
    print('Time ', str(int(time.time() - iter_time)), 's.')
    main_title = 'Test model. ' + "Loss: " + str(round(loss, 4)) + ", accuracy: " + str(
        round(accuracy, 4)) + ", epochs: " + str(epochs)
    if additional_title_info is not None:
        main_title = main_title + '\n' + additional_title_info

    y_test_prediction = model.predict(x_test)
    plot_helper.plot_result(y_test, y_test_prediction, 2, history, main_title, file_name)


if __name__ == '__main__':
    db_conn = db_access.create_db_connection(remote=False)
    df_list, sym_list = db_access.find_by_tickers_to_dateframe_parse_to_df_list(db_conn, [SELECTED_SYM],
                                                                                min_date=MIN_DATE, max_date=MAX_DATE)
    df = df_list[0]
    x_train, x_test, y_train, y_test = prepare_data(pca=.999)

    iter = 0
    param_grid = {
        'layers': [[1], [2], [3], [4], [5], [6], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [2, 2, 2], [3, 3, 3],
                   [4, 4, 4], [5, 5, 5], [6, 6, 6]]}
    grid = ParameterGrid(param_grid)
    for param in grid:
        layers = param['layers']
        title_info = 'Layers: [' + ''.join(str(e) + " " for e in layers) + ']'
        model = nn_model.create_seq_model(layers, input_size=x_train.shape[1], activation='relu',
                                          optimizer='adam', loss='binary_crossentropy', metric='binary_accuracy',
                                          output_neurons=1, overfitting_regularizer=0.005)
        run(model, x_train, x_test, y_train, y_test, epochs=30, batch_size=10, file_name="nn_model_" + str(iter),
            additional_title_info=title_info)
        iter = iter + 1
