import json
import os
import time

import benchmark_params
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from sklearn import model_selection
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import LabelEncoder

import csv_importer
import data_preprocessing
import nn_model
import plot_helper

SELECTED_SYM = 'GOOGL'
SAVE_MODEL_PATH = './../target/models/'
if not os.path.exists(SAVE_MODEL_PATH):
    os.makedirs(SAVE_MODEL_PATH)


def prepare_data(df, preprocessing_params):
    df, x, y = data_preprocessing.preprocess(df, preprocessing_params)

    if preprocessing_params.binary_classification:
        encoder = LabelEncoder()
        encoded_y = encoder.fit_transform(y)
    else:
        encoded_y = to_categorical(y)

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, encoded_y,
                                                                        test_size=preprocessing_params.test_size,
                                                                        shuffle=False)
    return x_train, x_test, y_train, y_test


def json_handler(Obj):
    if hasattr(Obj, 'jsonable'):
        return Obj.jsonable()
    else:
        return str(Obj)
        # raise TypeError('Object of type %s with value of %s is not JSON serializable' % (type(Obj), repr(Obj)))


def run(model, x_train, x_test, y_train, y_test, bench_params):
    learning_params = bench_params.learning_params
    model_params = bench_params.model_params
    iter_time = time.time()

    with open(SAVE_MODEL_PATH + 'config-' + learning_params.id + '.json', 'w') as outfile:
        json.dump(bench_params, outfile, default=json_handler, indent=True)

    earlyStopping = EarlyStopping(monitor='val_' + model_params.metric, min_delta=learning_params.early_stopping_min_delta,
                                  patience=learning_params.early_stopping_patience, verbose=0, mode='max',
                                  restore_best_weights=True)
    mcp_save = ModelCheckpoint(SAVE_MODEL_PATH + 'nn_weights-' + learning_params.id + '.hdf5', save_best_only=True,
                               monitor='val_' + model_params.metric,
                               mode='max')

    history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                        epochs=learning_params.epochs, batch_size=learning_params.batch_size,
                        callbacks=[earlyStopping, mcp_save], verbose=0)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    number_of_epochs_it_ran = len(history.history['loss'])

    print("Loss: ", loss, " Accuracy: ", accuracy, " epochs: ", number_of_epochs_it_ran)
    print('Time ', str(int(time.time() - iter_time)), 's.')
    main_title = 'Test model. ' + "Loss: " + str(round(loss, 4)) + ", accuracy: " + str(
        round(accuracy, 4)) + ", epochs: " + str(number_of_epochs_it_ran)
    main_title = main_title + '\n' + 'Layers: [' + ''.join(
        str(e) + " " for e in bench_params.model_params.layers) + ']'

    y_test_prediction = model.predict(x_test)
    plot_helper.plot_result(y_test, y_test_prediction, bench_params, history, main_title, 'nn-' + learning_params.id)


if __name__ == '__main__':
    df_list = csv_importer.import_data_from_files([SELECTED_SYM])
    df = df_list[0]

    bench_params = benchmark_params.default_params(binary_classification=True)
    bench_params.learning_params.epochs = 40

    param_grid = {
        'layers': [[2], [3]]
    }
    grid = ParameterGrid(param_grid)

    for param in grid:
        bench_params.update_from_dictionary(param)
        x_train, x_test, y_train, y_test = prepare_data(df, bench_params.preprocessing_params)
        model = nn_model.create_seq_model(x_train.shape[1], bench_params.model_params)
        run(model, x_train, x_test, y_train, y_test, bench_params)
