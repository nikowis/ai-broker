import json
import time

import keras
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.engine.saving import load_model
from sklearn.model_selection import ParameterGrid

import benchmark_data_preprocessing
import benchmark_file_helper
import benchmark_nn_model
import benchmark_params
import benchmark_plot_helper
import csv_importer
from benchmark_params import BenchmarkParams

CSV_TICKER = 'ticker'
CSV_ROC_AUC_COL = 'roc_auc'
CSV_ACC_COL = 'accuracy'
CSV_TRAIN_TIME_COL = 'train_time'
CSV_EPOCHS_COL = 'epochs'
CSV_ID_COL = 'ID'


def json_handler(Obj):
    if hasattr(Obj, 'jsonable'):
        return Obj.jsonable()
    else:
        return str(Obj)


class Benchmark:
    def __init__(self, symbols, bench_params: BenchmarkParams, changing_params_dict: dict) -> None:
        self.df_list, self.sym_list = csv_importer.import_data_from_files(symbols, bench_params.csv_files_dir)

        results_df = pd.DataFrame(
            data={CSV_ID_COL: [], CSV_EPOCHS_COL: [], CSV_TRAIN_TIME_COL: [], CSV_ACC_COL: [], CSV_ROC_AUC_COL: [],
                  CSV_TICKER: []})

        benchmark_file_helper.initialize_dirs(bench_params)

        grid = ParameterGrid(changing_params_dict)
        for param in grid:
            print('Parameters: {0}'.format(param))
            bench_params.update_from_dictionary(param)

            if bench_params.save_files:
                with open('{0}/config-{1}.json'.format(bench_params.save_model_path, bench_params.id), 'w') as outfile:
                    json.dump(bench_params, outfile, default=json_handler, indent=True)

            for symbol_it in range(0, len(symbols)):
                df = self.df_list[symbol_it]
                sym = self.sym_list[symbol_it]
                bench_params.curr_sym = sym
                df, x, y, x_train, x_test, y_train, y_test = benchmark_data_preprocessing.preprocess(df,
                                                                                                     bench_params)

                results_df = self.run(x_train, x_test, y_train, y_test, bench_params, results_df)

        if results_df is not None and len(results_df) > 0:
            results_df.to_csv('{0}/results.csv'.format(bench_params.save_model_path), index=False)

        print('Benchmark finished.')

    def run(self, x_train, x_test, y_train, y_test, bench_params, results_df):
        binary_classification = bench_params.binary_classification

        total_time = time.time()
        losses = []
        accuracies = []

        earlyStopping = EarlyStopping(monitor='val_' + bench_params.metric,
                                      min_delta=bench_params.early_stopping_min_delta,
                                      patience=bench_params.early_stopping_patience, verbose=0, mode='max',
                                      restore_best_weights=True)
        bench_params.curr_iter_num = 0
        minima_encountered = 0
        if bench_params.walk_forward_testing:
            input_size = x_train[0].shape[1]
        else:
            input_size = x_train.shape[1]
        while bench_params.curr_iter_num < bench_params.iterations:

            bench_params.curr_iter_num = bench_params.curr_iter_num + 1
            iter_start_time = time.time()

            model = benchmark_nn_model.create_seq_model(input_size, bench_params)

            callbacks = [earlyStopping]

            if bench_params.save_files:
                mcp_save = ModelCheckpoint(
                    benchmark_file_helper.get_model_path(bench_params), save_best_only=True,
                    monitor='val_' + bench_params.metric, mode='max')
                callbacks = [earlyStopping, mcp_save]

            model, history, accuracy, loss, y_test_prediction = self.learn(model, bench_params, callbacks,
                                                                           x_train,
                                                                           x_test,
                                                                           y_train, y_test)

            if (binary_classification and accuracy < 0.6) or ((not binary_classification) and accuracy < 0.45):
                if bench_params.verbose:
                    print('ID {0} iteration {1} encountered local minimum (accuracy {2}) retrying iteration...'.format(
                        bench_params.id, bench_params.curr_iter_num, round(accuracy, 4)))

                minima_encountered = minima_encountered + 1
                bench_params.curr_iter_num = bench_params.curr_iter_num - 1
                if minima_encountered > bench_params.iterations:
                    if bench_params.verbose:
                        print('ID {0}: encountering too many local minima - breaking infinite loop'.format(
                            bench_params.id))
                    return
                else:
                    continue
            losses.append(loss)
            accuracies.append(accuracy)
            number_of_epochs_it_ran = len(history.history['loss'])
            iter_time = time.time() - iter_start_time
            if bench_params.verbose:
                print('ID {0} iteration {1} of {2} loss {3} accuracy {4} epochs {5} time {6}'
                      .format(bench_params.id, bench_params.curr_iter_num, bench_params.iterations, round(loss, 4),
                              round(accuracy, 4),
                              number_of_epochs_it_ran, round(iter_time, 2)))

            main_title = 'Neural network model loss: {0}, accuracy {1}, epochs {2}\n hidden layers [{3}]'.format(
                round(loss, 4), round(accuracy, 4), number_of_epochs_it_ran, ''.join(
                    str(e) + " " for e in bench_params.layers))

            if bench_params.walk_forward_testing:
                concatenated_y_test = np.concatenate(y_test)
            else:
                concatenated_y_test = y_test
            fpr, tpr, roc_auc = benchmark_plot_helper.plot_result(concatenated_y_test, y_test_prediction, bench_params,
                                                                  history,
                                                                  main_title)

            results_df = results_df.append(
                {CSV_ID_COL: bench_params.id, CSV_EPOCHS_COL: int(number_of_epochs_it_ran),
                 CSV_TRAIN_TIME_COL: iter_time,
                 CSV_ACC_COL: accuracy,
                 CSV_ROC_AUC_COL: roc_auc,
                 CSV_TICKER: bench_params.curr_sym}, ignore_index=True)

        rounded_accuracy_mean = round(np.mean(accuracies), 4)
        rounded_loss_mean = round(np.mean(losses), 4)
        print(
            'ID {0} {1} avg loss {2} avg accuracy {3} total time {4} s.'.format(bench_params.id, bench_params.curr_sym, rounded_loss_mean,
                                                                                rounded_accuracy_mean,
                                                                                round(time.time() - total_time, 2)))
        if rounded_accuracy_mean > bench_params.satysfying_treshold:
            print('=============================================================================================')
        max_index = np.argmax(accuracies)
        benchmark_file_helper.copy_best_and_cleanup_files(bench_params, max_index,
                                                          round(max(accuracies), 4))

        return results_df

    def learn(self, model, bench_params: benchmark_params.NnBenchmarkParams, callbacks, x_train, x_test,
              y_train, y_test):
        if bench_params.walk_forward_testing:
            walk_iterations = len(x_train)
            walk_losses = []
            walk_accuracies = []
            walk_y_test_prediction = []
            walk_history = keras.callbacks.History()
            walk_history.history = {'loss': [], 'val_loss': [], bench_params.metric: [],
                                    'val_' + bench_params.metric: []}

            for walk_it in range(0, walk_iterations):
                walk_x_train = x_train[walk_it]
                walk_x_test = x_test[walk_it]
                walk_y_train = y_train[walk_it]
                walk_y_test = y_test[walk_it]
                if walk_it == 0:
                    epochs = bench_params.epochs
                else:
                    epochs = bench_params.walk_forward_retrain_epochs

                history = model.fit(walk_x_train, walk_y_train, validation_data=(walk_x_test, walk_y_test),
                                    epochs=epochs, batch_size=bench_params.batch_size,
                                    callbacks=callbacks, verbose=0)
                walk_history.history['loss'] += history.history['loss']
                walk_history.history['val_loss'] += history.history['val_loss']
                walk_history.history[bench_params.metric] += history.history[bench_params.metric]
                walk_history.history['val_' + bench_params.metric] += history.history[
                    'val_' + bench_params.metric]

                ls, acc = model.evaluate(walk_x_test, walk_y_test, verbose=0)
                y_test_prediction = model.predict(walk_x_test)
                walk_losses.append(ls)
                walk_accuracies.append(acc)
                walk_y_test_prediction += y_test_prediction.tolist()
            if bench_params.verbose:
                print('Walk accuracies: [{0}]'.format(walk_accuracies))
            return model, walk_history, np.mean(walk_accuracies), np.mean(walk_losses), np.array(walk_y_test_prediction)
        else:
            history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                                epochs=bench_params.epochs, batch_size=bench_params.batch_size,
                                callbacks=callbacks, verbose=0)
            if bench_params.save_files:
                # restores best epoch of this iteration
                model = load_model(benchmark_file_helper.get_model_path(bench_params))
            loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
            y_test_prediction = model.predict(x_test)
            return model, history, accuracy, loss, y_test_prediction


if __name__ == '__main__':
    bench_params = benchmark_params.NnBenchmarkParams(binary_classification=True)
    bench_params.iterations = 2

    bench = Benchmark(['GOOGL', 'AMZN'], bench_params, {'epochs': [5],
                                                        'layers': [[]]})
