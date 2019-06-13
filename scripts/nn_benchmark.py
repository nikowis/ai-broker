import json
import os
import time
from shutil import copyfile

import keras
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
import stock_constants as const

SATYSFYING_TRESHOLD = 0.85
ROC_AUC_COL = 'roc_auc'
ACC_COL = 'accuracy'
TRAIN_TIME_COL = 'train_time'
EPOCHS_COL = 'epochs'
ID_COL = 'ID'

SELECTED_SYM = 'GOOGL'
SAVE_MODEL_PATH = const.TARGET_DIR + '/models/'
if not os.path.exists(SAVE_MODEL_PATH):
    os.makedirs(SAVE_MODEL_PATH)


def json_handler(Obj):
    if hasattr(Obj, 'jsonable'):
        return Obj.jsonable()
    else:
        return str(Obj)


def run(x_train, x_test, y_train, y_test, bench_params, results_df: pd.DataFrame, verbose=True):
    learning_params = bench_params.learning_params
    model_params = bench_params.model_params
    binary_classification = bench_params.preprocessing_params.binary_classification

    total_time = time.time()
    losses = []
    accuracies = []
    best_model_paths = []

    with open('{0}config-{1}.json'.format(SAVE_MODEL_PATH, learning_params.id), 'w') as outfile:
        json.dump(bench_params, outfile, default=json_handler, indent=True)

    earlyStopping = EarlyStopping(monitor='val_' + model_params.metric,
                                  min_delta=learning_params.early_stopping_min_delta,
                                  patience=learning_params.early_stopping_patience, verbose=0, mode='max',
                                  restore_best_weights=True)
    curr_iter_num = 0
    minima_encountered = 0
    if learning_params.walk_forward_testing:
        input_size = x_train[0].shape[1]
    else:
        input_size = x_train.shape[1]
    while curr_iter_num < learning_params.iterations:

        curr_iter_num = curr_iter_num + 1
        iter_start_time = time.time()

        model = nn_model.create_seq_model(input_size, bench_params.model_params)

        model_path = '{0}nn_weights-{1}-{2}.hdf5'.format(SAVE_MODEL_PATH, learning_params.id, curr_iter_num)
        best_model_paths.append(model_path)
        mcp_save = ModelCheckpoint(
            model_path,
            save_best_only=True,
            monitor='val_' + model_params.metric,
            mode='max')

        callbacks = [earlyStopping, mcp_save]
        model, history, accuracy, loss, y_test_prediction = learn(model, bench_params, callbacks, model_path, x_train, x_test,
                                               y_train, y_test, verbose)

        if (binary_classification and accuracy < 0.6) or ((not binary_classification) and accuracy < 0.45):
            if verbose:
                print('ID {0} iteration {1} encountered local minimum (accuracy {2}) retrying iteration...'.format(
                    learning_params.id, curr_iter_num, round(accuracy, 4)))

            minima_encountered = minima_encountered + 1
            curr_iter_num = curr_iter_num - 1
            if minima_encountered > learning_params.iterations:
                if verbose:
                    print('ID {0}: encountering too many local minima - breaking infinite loop'.format(learning_params.id))
                return
            else:
                continue
        losses.append(loss)
        accuracies.append(accuracy)
        number_of_epochs_it_ran = len(history.history['loss'])
        iter_time = time.time() - iter_start_time
        if verbose:
            print('ID {0} iteration {1} of {2} loss {3} accuracy {4} epochs {5} time {6}'
                  .format(learning_params.id, curr_iter_num, learning_params.iterations, round(loss, 4), round(accuracy, 4),
                          number_of_epochs_it_ran, round(iter_time, 2)))

        main_title = 'Neural network model loss: {0}, accuracy {1}, epochs {2}\n hidden layers [{3}]'.format(
            round(loss, 4), round(accuracy, 4), number_of_epochs_it_ran, ''.join(
                str(e) + " " for e in bench_params.model_params.layers))

        if learning_params.walk_forward_testing:
            y_test = np.concatenate(y_test)

        fpr, tpr, roc_auc = plot_helper.plot_result(y_test, y_test_prediction, bench_params, history, main_title,
                                                        'nn-{0}-{1}'.format(learning_params.id, curr_iter_num))

        results_df = results_df.append(
            {ID_COL: learning_params.id, EPOCHS_COL: int(number_of_epochs_it_ran), TRAIN_TIME_COL: iter_time,
             ACC_COL: accuracy,
             ROC_AUC_COL: roc_auc}, ignore_index=True)

    rounded_accuracy_mean = round(np.mean(accuracies), 4)
    rounded_loss_mean = round(np.mean(losses), 4)
    print('ID {0} avg loss {1} avg accuracy {2} total time {3} s.'.format(learning_params.id, rounded_loss_mean,
                                                                          rounded_accuracy_mean,
                                                                          round(time.time() - total_time, 2)))
    if rounded_accuracy_mean > SATYSFYING_TRESHOLD:
        print('=============================================================================================')
    max_index = np.argmax(accuracies)
    best_model_of_all_path = best_model_paths[max_index]
    copyfile(best_model_of_all_path,
             '{0}nn_weights-{1}-best-accuracy-{2}.hdf5'.format(SAVE_MODEL_PATH, learning_params.id,
                                                               round(max(accuracies), 4)))
    return results_df


def learn(model, bench_params, callbacks, model_path, x_train, x_test, y_train, y_test, verbose=True):
    learning_params = bench_params.learning_params
    if learning_params.walk_forward_testing:
        walk_iterations = len(x_train)
        walk_losses = []
        walk_accuracies = []
        walk_y_test_prediction = []
        walk_history = keras.callbacks.History()
        walk_history.history = { 'loss':[], 'val_loss': [], bench_params.model_params.metric: [],
                                 'val_' + bench_params.model_params.metric:[]  }

        for walk_it in range(0, walk_iterations):
            walk_x_train = x_train[walk_it]
            walk_x_test = x_test[walk_it]
            walk_y_train = y_train[walk_it]
            walk_y_test = y_test[walk_it]
            if walk_it == 0:
                epochs = learning_params.epochs
            else:
                epochs = learning_params.walk_forward_retrain_epochs

            history = model.fit(walk_x_train, walk_y_train, validation_data=(walk_x_test, walk_y_test),
                                epochs=epochs, batch_size=learning_params.batch_size,
                                callbacks=callbacks, verbose=0)
            walk_history.history['loss'] += history.history['loss']
            walk_history.history['val_loss'] += history.history['val_loss']
            walk_history.history[bench_params.model_params.metric] += history.history[bench_params.model_params.metric]
            walk_history.history['val_' + bench_params.model_params.metric] += history.history['val_' + bench_params.model_params.metric]

            ls, acc = model.evaluate(walk_x_test, walk_y_test, verbose=0)
            y_test_prediction = model.predict(walk_x_test)
            walk_losses.append(ls)
            walk_accuracies.append(acc)
            walk_y_test_prediction += y_test_prediction.tolist()
        if verbose:
            print('Walk accuracies: [{0}]'.format(walk_accuracies))
        return model, walk_history, np.mean(walk_accuracies), np.mean(walk_losses), np.array(walk_y_test_prediction)
    else:
        history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                            epochs=learning_params.epochs, batch_size=learning_params.batch_size,
                            callbacks=callbacks, verbose=0)
        # restores best epoch of this iteration
        model = load_model(model_path)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        y_test_prediction = model.predict(x_test)
        return model, history, accuracy, loss, y_test_prediction


if __name__ == '__main__':
    df_list = csv_importer.import_data_from_files([SELECTED_SYM])
    df = df_list[0]

    bench_params = benchmark_params.default_params(binary_classification=True)
    bench_params.learning_params.iterations=1
    bench_params.learning_params.walk_forward_testing=True
    bench_params.preprocessing_params.walk_forward_testing=True

    param_grid = {
        'epochs': [10, 20, 30, 40, 50, 70],
        'layers': [[], [2], [5]],
        'walk_forward_retrain_epochs':[1, 2, 3, 5, 10],
        'walk_forward_max_train_window_size':[None, 2000, 1000, 500, 300, 150],
        'walk_forward_test_window_size':[20, 15 ,10]
    }
    grid = ParameterGrid(param_grid)

    results_df = pd.DataFrame(data={ID_COL: [], EPOCHS_COL: [], TRAIN_TIME_COL: [], ACC_COL: [], ROC_AUC_COL: []})

    for param in grid:
        print('Parameters: {0}'.format(param))
        bench_params.update_from_dictionary(param)
        result_df_custom_param_val = '_'.join(str(v) for k, v in param.items())

        df, x, y, x_train, x_test, y_train, y_test = data_preprocessing.preprocess(df,
                                                                                   bench_params.preprocessing_params)

        results_df = run(x_train, x_test, y_train, y_test, bench_params, results_df, verbose=False)


    if results_df is not None and len(results_df) > 0:
        results_df.to_csv('{0}results.csv'.format(SAVE_MODEL_PATH), index=False)

    print('Program finished.')