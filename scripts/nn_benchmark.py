import json
import os
import time
from shutil import copyfile

import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from sklearn.model_selection import ParameterGrid

import benchmark_params
import csv_importer
import data_preprocessing
import nn_model
import plot_helper

SELECTED_SYM = 'GOOGL'
SAVE_MODEL_PATH = './../target/models/'
if not os.path.exists(SAVE_MODEL_PATH):
    os.makedirs(SAVE_MODEL_PATH)


def json_handler(Obj):
    if hasattr(Obj, 'jsonable'):
        return Obj.jsonable()
    else:
        return str(Obj)


def run(x_train, x_test, y_train, y_test, bench_params):
    learning_params = bench_params.learning_params
    model_params = bench_params.model_params
    binary_classification = bench_params.preprocessing_params.binary_classification

    total_time = time.time()
    losses = []
    accuracies = []
    best_model_paths = []

    with open(SAVE_MODEL_PATH + 'config-' + learning_params.id + '.json', 'w') as outfile:
        json.dump(bench_params, outfile, default=json_handler, indent=True)

    earlyStopping = EarlyStopping(monitor='val_' + model_params.metric,
                                  min_delta=learning_params.early_stopping_min_delta,
                                  patience=learning_params.early_stopping_patience, verbose=0, mode='max',
                                  restore_best_weights=True)
    curr_iter_num = 0
    minima_encountered = 0
    while curr_iter_num < learning_params.iterations:

        curr_iter_num = curr_iter_num + 1
        iter_time = time.time()

        model = nn_model.create_seq_model(x_train.shape[1], bench_params.model_params)

        model_path = SAVE_MODEL_PATH + 'nn_weights-' + learning_params.id + '-' + str(curr_iter_num) + '.hdf5'
        best_model_paths.append(model_path)
        mcp_save = ModelCheckpoint(
            model_path,
            save_best_only=True,
            monitor='val_' + model_params.metric,
            mode='max')

        history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                            epochs=learning_params.epochs, batch_size=learning_params.batch_size,
                            callbacks=[earlyStopping, mcp_save], verbose=0)

        # restores best epoch of this iteration
        model = load_model(model_path)

        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        if (binary_classification and accuracy < 0.6) or ((not binary_classification) and accuracy < 0.45):
            print("ID ", learning_params.id, " iteration ", str(curr_iter_num), 'encountered local minimum (accuracy',
                  str(round(accuracy, 4)), ' ) retrying iteration...')
            minima_encountered = minima_encountered + 1
            curr_iter_num = curr_iter_num - 1
            if minima_encountered > learning_params.iterations:
                print("ID ", learning_params.id, "encountering too many local minima - breaking infinite loop ")
                return
            else:
                continue
        losses.append(loss)
        accuracies.append(accuracy)
        number_of_epochs_it_ran = len(history.history['loss'])
        print("ID ", learning_params.id, " iteration ", str(curr_iter_num), "of", str(learning_params.iterations), "id",
              learning_params.id, "loss:", str(round(loss, 4)), "accuracy:",
              str(round(accuracy, 4)), "epochs:", number_of_epochs_it_ran, "time ",
              str(int(time.time() - iter_time)), 's.')
        main_title = 'Test model. ' + "Loss: " + str(round(loss, 4)) + ", accuracy: " + str(
            round(accuracy, 4)) + ", epochs: " + str(number_of_epochs_it_ran)
        main_title = main_title + '\n' + 'Layers: [' + ''.join(
            str(e) + " " for e in bench_params.model_params.layers) + ']'

        y_test_prediction = model.predict(x_test)
        plot_helper.plot_result(y_test, y_test_prediction, bench_params, history, main_title,
                                'nn-' + learning_params.id + '-' + str(curr_iter_num))
    print("ID ", learning_params.id, "avg loss:", str(round(np.mean(losses), 4)), "avg accuracy:",
          str(round(np.mean(accuracies), 4)), "total time ", str(int(time.time() - total_time)), 's.')
    max_index = np.argmax(accuracies)
    best_model_of_all_path = best_model_paths[max_index]
    copyfile(best_model_of_all_path, SAVE_MODEL_PATH + 'nn_weights-' + learning_params.id + '.hdf5')


if __name__ == '__main__':
    df_list = csv_importer.import_data_from_files([SELECTED_SYM])
    df = df_list[0]

    bench_params = benchmark_params.default_params(binary_classification=True)
    bench_params.learning_params.epochs = 50
    # bench_params.model_params.regularizer = .01
    bench_params.learning_params.iterations = 2
    param_grid = {
        'layers': [[]]
    }
    grid = ParameterGrid(param_grid)

    for param in grid:
        bench_params.update_from_dictionary(param)
        df, x, y, x_train, x_test, y_train, y_test = data_preprocessing.preprocess(df,
                                                                                   bench_params.preprocessing_params)

        run(x_train, x_test, y_train, y_test, bench_params)
