import time
import uuid

from sklearn import model_selection
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import LabelEncoder

import benchmark_params
import csv_importer
import data_preprocessing
import nn_model
import plot_helper

SELECTED_SYM = 'GOOGL'


def prepare_data(df, preprocessing_params):
    df, x, y = data_preprocessing.preprocess(df, preprocessing_params)
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_Y = encoder.transform(y)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, encoded_Y,
                                                                        test_size=preprocessing_params.test_size,
                                                                        shuffle=False)
    return x_train, x_test, y_train, y_test


def run(model, x_train, x_test, y_train, y_test, benchmark_params, file_name="test_model_" + uuid.uuid4().hex):
    learning_params = benchmark_params.learning_params
    iter_time = time.time()
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                        epochs=learning_params.epochs, batch_size=learning_params.batch_size, verbose=0)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print("Loss: ", loss, " Accuracy: ", accuracy, " epochs: ", learning_params.epochs)
    print('Time ', str(int(time.time() - iter_time)), 's.')
    main_title = 'Test model. ' + "Loss: " + str(round(loss, 4)) + ", accuracy: " + str(
        round(accuracy, 4)) + ", epochs: " + str(learning_params.epochs)
    main_title = main_title + '\n' + 'Layers: [' + ''.join(str(e) + " " for e in benchmark_params.model_params.layers) + ']'

    y_test_prediction = model.predict(x_test)
    plot_helper.plot_result(y_test, y_test_prediction, 2, history, main_title, file_name)


if __name__ == '__main__':
    df_list = csv_importer.import_data_from_files([SELECTED_SYM])
    df = df_list[0]

    benchmark_params = benchmark_params.default_params(True)
    iter = 0
    param_grid = {
        'layers': [[1], [2], [3], [4], [5], [6], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [2, 2, 2], [3, 3, 3],
                   [4, 4, 4], [5, 5, 5], [6, 6, 6]]}
    grid = ParameterGrid(param_grid)

    for param in grid:
        benchmark_params.update_from_dictionary(param)

        x_train, x_test, y_train, y_test = prepare_data(df, benchmark_params.preprocessing_params)

        model = nn_model.create_seq_model(x_train.shape[1], benchmark_params.model_params)

        run(model, x_train, x_test, y_train, y_test, benchmark_params, file_name="nn_model_" + str(iter))

        iter = iter + 1
