import json
import os
import time
from shutil import copyfile

import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from sklearn.model_selection import ParameterGrid

import benchmark_params
import csv_importer
import data_preprocessing
import nn_model
import plot_helper

ROC_AUC_COL = 'roc_auc'
ACC_COL = 'accuracy'
TRAIN_TIME_COL = 'train_time'
EPOCHS_COL = 'epochs'
ID_COL = 'ID'

SELECTED_SYM = 'GOOGL'
SAVE_MODEL_PATH = './../target/models/'
if not os.path.exists(SAVE_MODEL_PATH):
    os.makedirs(SAVE_MODEL_PATH)


def json_handler(Obj):
    if hasattr(Obj, 'jsonable'):
        return Obj.jsonable()
    else:
        return str(Obj)


def run(x_train, x_test, y_train, y_test, bench_params, results_df: pd.DataFrame):
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
        iter_start_time = time.time()

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
        iter_time = time.time() - iter_start_time
        print("ID ", learning_params.id, " iteration ", str(curr_iter_num), "of", str(learning_params.iterations), "id",
              learning_params.id, "loss:", str(round(loss, 4)), "accuracy:",
              str(round(accuracy, 4)), "epochs:", number_of_epochs_it_ran, "time ",
              str(round(iter_time, 2)), 's.')
        main_title = 'Test model. ' + "Loss: " + str(round(loss, 4)) + ", accuracy: " + str(
            round(accuracy, 4)) + ", epochs: " + str(number_of_epochs_it_ran)
        main_title = main_title + '\n' + 'Layers: [' + ''.join(
            str(e) + " " for e in bench_params.model_params.layers) + ']'

        y_test_prediction = model.predict(x_test)
        fpr, tpr, roc_auc = plot_helper.plot_result(y_test, y_test_prediction, bench_params, history, main_title,
                                                    'nn-' + learning_params.id + '-' + str(curr_iter_num))

        results_df = results_df.append(
            {ID_COL: learning_params.id, EPOCHS_COL: int(number_of_epochs_it_ran), TRAIN_TIME_COL: iter_time, ACC_COL: accuracy,
             ROC_AUC_COL: roc_auc}, ignore_index=True)

    print("ID ", learning_params.id, "avg loss:", str(round(np.mean(losses), 4)), "avg accuracy:",
          str(round(np.mean(accuracies), 4)), "total time ", str(round(time.time() - total_time, 2)), 's.')
    max_index = np.argmax(accuracies)
    best_model_of_all_path = best_model_paths[max_index]
    copyfile(best_model_of_all_path, SAVE_MODEL_PATH + 'nn_weights-' + learning_params.id + '.hdf5')
    return results_df


if __name__ == '__main__':
    df_list = csv_importer.import_data_from_files([SELECTED_SYM])
    df = df_list[0]

    bench_params = benchmark_params.default_params(binary_classification=False)
    bench_params.learning_params.epochs = 10
    # bench_params.model_params.regularizer = .01
    bench_params.learning_params.iterations = 2
    param_grid = {
        'layers': [[], [1], [2], [3]],
    }
    grid = ParameterGrid(param_grid)

    results_df = pd.DataFrame(data={ID_COL: [], EPOCHS_COL: [], TRAIN_TIME_COL: [], ACC_COL: [], ROC_AUC_COL: []})

    for param in grid:
        bench_params.update_from_dictionary(param)
        result_df_custom_param_val = '_'.join(str(v) for k, v in param.items())

        df, x, y, x_train, x_test, y_train, y_test = data_preprocessing.preprocess(df,
                                                                                   bench_params.preprocessing_params)

        results_df = run(x_train, x_test, y_train, y_test, bench_params, results_df)

    results_df.to_csv(SAVE_MODEL_PATH + 'results.csv', index=False)

