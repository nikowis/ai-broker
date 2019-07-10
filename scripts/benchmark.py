import json
import time

import numpy as np
import pandas as pd
from keras.engine.saving import load_model
from sklearn.model_selection import ParameterGrid

import benchmark_data_preprocessing
import benchmark_file_helper
import benchmark_plot_helper
import benchmark_roc_auc
import csv_importer
import stock_constants
from benchmark_params import BenchmarkParams

CSV_TICKER = 'ticker'
CSV_ROC_AUC_COL = 'roc_auc'
CSV_ACC_COL = 'accuracy'
CSV_TRAIN_TIME_COL = 'train_time'
CSV_ID_COL = 'ID'


def json_handler(Obj):
    if hasattr(Obj, 'jsonable'):
        return Obj.jsonable()
    else:
        return str(Obj)


class Benchmark:
    def __init__(self, symbols, bench_params: BenchmarkParams, changing_params_dict: dict = None) -> None:
        self.bench_params = bench_params
        benchmark_time = time.time()
        self.df_list, self.sym_list = csv_importer.import_data_from_files(symbols, bench_params.csv_files_path)
        self.results_df = pd.DataFrame()
        benchmark_file_helper.initialize_dirs(bench_params)
        if changing_params_dict is not None:
            grid = ParameterGrid(changing_params_dict)
            for param in grid:
                print('Parameters: {0}'.format(param))
                bench_params.update_from_dictionary(param)
                self.run(symbols)
        else:
            self.run(symbols)

        benchmark_file_helper.save_results(self.results_df, bench_params)

        if bench_params.examined_params is not None:
            split_params = bench_params.examined_params.split(',')
            for examined_param in split_params:
                self.results_df[examined_param] = self.results_df[examined_param].astype(str)
            mean_groupby = self.results_df.groupby(split_params).mean()
            print('Examination of {0}: {1}'.format(bench_params.examined_params, mean_groupby))

        print('Benchmark finished in {0}.'.format(round(time.time() - benchmark_time, 2)))

    def run(self, symbols):
        bench_params = self.bench_params
        if bench_params.save_files:
            with open('{0}/config-{1}.json'.format(bench_params.save_model_path, bench_params.id), 'w') as outfile:
                json.dump(bench_params, outfile, default=json_handler, indent=True)
        for symbol_it in range(0, len(symbols)):
            df = self.df_list[symbol_it]
            sym = self.sym_list[symbol_it]
            bench_params.curr_sym = sym
            x, y, x_train, x_test, y_train, y_test, _, _ = benchmark_data_preprocessing.preprocess(df,
                                                                                             bench_params)

            self.run_single_company(x_train, x_test, y_train, y_test)

    def run_single_company(self, x_train, x_test, y_train, y_test):
        bench_params = self.bench_params
        total_time = time.time()
        accuracies = []
        roc_auc_values = []

        bench_params.curr_iter_num = 0
        minima_encountered = 0
        if bench_params.walk_forward_testing:
            bench_params.input_size = x_train[0].shape[1]
        else:
            bench_params.input_size = x_train.shape[1]
        while bench_params.curr_iter_num < bench_params.iterations:

            bench_params.curr_iter_num = bench_params.curr_iter_num + 1
            iter_start_time = time.time()

            model = self.create_model()

            callbacks = self.create_callbacks()

            model, accuracy, loss, fpr, tpr, roc_auc, y_test_prediction, history = self.learn_and_evaluate(model,
                                                                                                           callbacks,
                                                                                                           x_train,
                                                                                                           x_test,
                                                                                                           y_train,
                                                                                                           y_test)

            if bench_params.binary_classification:
                roc_auc_value = roc_auc
            else:
                roc_auc_value = roc_auc[stock_constants.MICRO_ROC_KEY]

            if (bench_params.binary_classification and roc_auc_value < 0.5) or (
                    (not bench_params.binary_classification) and roc_auc_value < 0.5):
                if bench_params.verbose:
                    print('ID {0} iteration {1} encountered local minimum (auc {2}) retrying iteration...'.format(
                        bench_params.id, bench_params.curr_iter_num, round(roc_auc_value, 4)))

                minima_encountered = minima_encountered + 1
                bench_params.curr_iter_num = bench_params.curr_iter_num - 1
                if minima_encountered > bench_params.iterations:
                    if bench_params.verbose:
                        print('ID {0}: encountering too many local minima - breaking infinite loop'.format(
                            bench_params.id))
                else:
                    continue
            accuracies.append(accuracy)
            roc_auc_values.append(roc_auc_value)
            if history is not None:
                number_of_epochs_it_ran = len(history.history['loss'])
            else:
                number_of_epochs_it_ran = None
            iter_time = time.time() - iter_start_time
            if bench_params.verbose:
                print('ID {0} {1} iteration {2} of {3} accuracy {4} roc_auc {5} epochs {6} time {7}'
                      .format(bench_params.id, bench_params.curr_sym, bench_params.curr_iter_num,
                              bench_params.iterations, round(accuracy, 4),
                              round(roc_auc_value, 4),
                              number_of_epochs_it_ran, round(iter_time, 2)))

            main_title = 'Neural network model accuracy: {0}, roc_auc {1}, epochs {2}, company {3}\n'.format(
                round(accuracy, 4), round(roc_auc_value, 4), number_of_epochs_it_ran, bench_params.curr_sym)
            if bench_params.examined_params is not None and bench_params.examined_params != '':
                split_params = bench_params.examined_params.split(',')
                for param in split_params:
                    main_title = main_title + ' {0}:{1} '.format(param,
                                                                 getattr(bench_params, param, ''))
            if bench_params.save_files:
                if bench_params.walk_forward_testing:
                    concatenated_y_test = np.concatenate(y_test)
                else:
                    concatenated_y_test = y_test
                benchmark_plot_helper.plot_result(concatenated_y_test, y_test_prediction, bench_params,
                                                  history, fpr, tpr, roc_auc,
                                                  main_title)

            result_dict = {CSV_ID_COL: bench_params.id,
                           CSV_TRAIN_TIME_COL: iter_time, CSV_ACC_COL: accuracy, CSV_ROC_AUC_COL: roc_auc_value,
                           CSV_TICKER: bench_params.curr_sym}

            if bench_params.examined_params is not None:
                examined_params = bench_params.examined_params.split(',')
                for examined_param in examined_params:
                    result_dict.update({examined_param: getattr(bench_params, examined_param, None)})

            self.results_df = self.results_df.append(result_dict, ignore_index=True)

        rounded_roc_auc_mean = round(np.mean(roc_auc_values), 4)
        rounded_acc_mean = round(np.mean(accuracies), 4)
        print(
            'ID {0} {1} avg accuracy {2} avg roc_auc {3} total time {4} s.'.format(bench_params.id,
                                                                                   bench_params.curr_sym,
                                                                                   rounded_acc_mean,
                                                                                   rounded_roc_auc_mean,
                                                                                   round(time.time() - total_time, 2)))
        if rounded_roc_auc_mean > bench_params.satysfying_treshold:
            print('=============================================================================================')
        max_index = np.argmax(accuracies)
        benchmark_file_helper.copy_best_and_cleanup_files(bench_params, max_index,
                                                          round(max(accuracies), 4))

    def learn_and_evaluate(self, model, callbacks, x_train, x_test,
                           y_train, y_test):
        bench_params = self.bench_params
        if bench_params.walk_forward_testing:
            walk_iterations = len(x_train)
            walk_losses = []
            walk_accuracies = []
            walk_y_test_prediction = []
            walk_history = self.create_history_object()

            for walk_it in range(0, walk_iterations):
                walk_x_train = x_train[walk_it]
                walk_x_test = x_test[walk_it]
                walk_y_train = y_train[walk_it]
                walk_y_test = y_test[walk_it]

                bench_params.input_size = walk_x_train.shape[1]

                model = self.create_model()
                model, history = self.fit_model(model, callbacks, walk_x_train, walk_y_train, walk_x_test,
                                                walk_y_test)
                self.update_walk_history(history, walk_history)

                acc, ls, y_test_prediction = self.evaluate_predict(model, walk_x_test, walk_y_test)
                walk_losses.append(ls)
                walk_accuracies.append(acc)
                walk_y_test_prediction += y_test_prediction.tolist()

            if bench_params.verbose:
                print('Walk accuracies: [{0}]'.format(walk_accuracies))

            roc_y_test = np.concatenate(y_test)
            accuracy = np.mean(walk_accuracies)
            loss = np.mean(walk_losses)
            y_test_prediction = np.array(walk_y_test_prediction)
            history = walk_history

        else:
            model, history = self.fit_model(model, callbacks, x_train, y_train, x_test, y_test)
            if bench_params.save_files and bench_params.save_model:
                # restores best epoch of this iteration
                model = load_model(benchmark_file_helper.get_model_path(bench_params))
            accuracy, loss, y_test_prediction = self.evaluate_predict(model, x_test, y_test)
            roc_y_test = y_test

        fpr, tpr, roc_auc = benchmark_roc_auc.calculate_roc_auc(roc_y_test, y_test_prediction,
                                                                bench_params.classes_count,
                                                                bench_params.one_hot_encode_labels)

        return model, accuracy, loss, fpr, tpr, roc_auc, y_test_prediction, history

    def create_model(self):
        """Create predicting model, return model"""
        return None

    def create_callbacks(self):
        """Create callbacks used while learning"""
        return None

    def evaluate_predict(self, model, x_test, y_test):
        """Evaluate on test data, predict labels for x_test, return (accuracy, loss, y_prediction)"""
        return None, None, None

    def fit_model(self, model, callbacks, x_train, y_train, x_test, y_test, epochs=None):
        """Fit model on train data, return model and learning history or none"""
        return None, None

    def update_walk_history(self, history, walk_history):
        """Update history object with walk forward learning history"""
        return None

    def create_history_object(self):
        """Create an empty history object for walk forward learning"""
        return None
