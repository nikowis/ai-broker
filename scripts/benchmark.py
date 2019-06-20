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
import benchmark_roc_auc
import csv_importer
import stock_constants
from benchmark_params import BenchmarkParams, NnBenchmarkParams

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
    def __init__(self, symbols, bench_params: BenchmarkParams, changing_params_dict: dict) -> None:
        benchmark_time = time.time()
        self.df_list, self.sym_list = csv_importer.import_data_from_files(symbols, bench_params.csv_files_path)
        results_df = pd.DataFrame()
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
                x, y, x_train, x_test, y_train, y_test = benchmark_data_preprocessing.preprocess(df,
                                                                                                     bench_params)
                if bench_params.walk_forward_testing:
                    bench_params.input_size = x_train[0].shape[1]
                else:
                    bench_params.input_size = x_train.shape[1]

                results_df = self.run(x_train, x_test, y_train, y_test, bench_params, results_df)

        benchmark_file_helper.save_results(results_df, bench_params)

        if bench_params.examined_params is not None:
            split_params = bench_params.examined_params.split(',')
            for examined_param in split_params:
                results_df[examined_param] = results_df[examined_param].astype(str)
            mean_groupby = results_df.groupby(split_params).mean()
            print('Examination of {0}: {1}'.format(bench_params.examined_params, mean_groupby))

        print('Benchmark finished in {0}.'.format(round(time.time() - benchmark_time, 2)))

    def run(self, x_train, x_test, y_train, y_test, bench_params, results_df):

        total_time = time.time()
        losses = []
        roc_auc_values = []

        bench_params.curr_iter_num = 0
        minima_encountered = 0

        while bench_params.curr_iter_num < bench_params.iterations:

            bench_params.curr_iter_num = bench_params.curr_iter_num + 1
            iter_start_time = time.time()

            model = self.create_model(bench_params)

            callbacks = self.create_callbacks(bench_params)

            model, accuracy, loss, fpr, tpr, roc_auc, y_test_prediction, history = self.learn_and_evaluate(model,
                                                                                                           bench_params,
                                                                                                           callbacks,
                                                                                                           x_train,
                                                                                                           x_test,
                                                                                                           y_train,
                                                                                                           y_test)

            if bench_params.binary_classification:
                roc_auc_value = roc_auc
            else:
                roc_auc_value = roc_auc[stock_constants.MICRO_ROC_KEY]

            if (bench_params.binary_classification and roc_auc_value < 0.7) or (
                    (not bench_params.binary_classification) and accuracy < 0.65):
                if bench_params.verbose:
                    print('ID {0} iteration {1} encountered local minimum (auc {2}) retrying iteration...'.format(
                        bench_params.id, bench_params.curr_iter_num, round(roc_auc_value, 4)))

                minima_encountered = minima_encountered + 1
                bench_params.curr_iter_num = bench_params.curr_iter_num - 1
                if minima_encountered > bench_params.iterations:
                    if bench_params.verbose:
                        print('ID {0}: encountering too many local minima - breaking infinite loop'.format(
                            bench_params.id))
                    return results_df
                else:
                    continue
            losses.append(loss)
            roc_auc_values.append(roc_auc_value)
            number_of_epochs_it_ran = len(history.history['loss'])
            iter_time = time.time() - iter_start_time
            if bench_params.verbose:
                print('ID {0} {1} iteration {2} of {3} loss {4} roc_auc {5} epochs {6} time {7}'
                      .format(bench_params.id, bench_params.curr_sym, bench_params.curr_iter_num,
                              bench_params.iterations, round(loss, 4),
                              round(roc_auc_value, 4),
                              number_of_epochs_it_ran, round(iter_time, 2)))

            main_title = 'Neural network model loss: {0}, roc_auc {1}, epochs {2}\n hidden layers [{3}] company {4} examined param {5}:{6}'.format(
                round(loss, 4), round(roc_auc_value, 4), number_of_epochs_it_ran, ''.join(
                    str(e) + " " for e in bench_params.layers), bench_params.curr_sym,
                bench_params.examined_params.split(',')[0],
                getattr(bench_params, bench_params.examined_params.split(',')[0], ''))

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

            results_df = results_df.append(
                result_dict, ignore_index=True)

        rounded_roc_auc_mean = round(np.mean(roc_auc_values), 4)
        rounded_loss_mean = round(np.mean(losses), 4)
        print(
            'ID {0} {1} avg loss {2} avg roc_auc {3} total time {4} s.'.format(bench_params.id, bench_params.curr_sym,
                                                                               rounded_loss_mean,
                                                                               rounded_roc_auc_mean,
                                                                               round(time.time() - total_time, 2)))
        if rounded_roc_auc_mean > bench_params.satysfying_treshold:
            print('=============================================================================================')
        max_index = np.argmax(roc_auc_values)
        benchmark_file_helper.copy_best_and_cleanup_files(bench_params, max_index,
                                                          round(max(roc_auc_values), 4))

        return results_df

    def learn_and_evaluate(self, model, bench_params: benchmark_params.NnBenchmarkParams, callbacks, x_train, x_test,
                           y_train, y_test):
        if bench_params.walk_forward_testing:
            walk_iterations = len(x_train)
            walk_losses = []
            walk_accuracies = []
            walk_y_test_prediction = []
            walk_history = self.create_history_object(bench_params)

            for walk_it in range(0, walk_iterations):
                walk_x_train = x_train[walk_it]
                walk_x_test = x_test[walk_it]
                walk_y_train = y_train[walk_it]
                walk_y_test = y_test[walk_it]
                if walk_it == 0:
                    epochs = bench_params.epochs
                else:
                    epochs = bench_params.walk_forward_retrain_epochs

                if bench_params.walk_forward_learn_from_scratch:
                    epochs = bench_params.epochs
                    model = self.create_model(bench_params)

                history = self.fit_model(bench_params, model, callbacks, walk_x_train, walk_y_train, walk_x_test,
                                         walk_y_test, epochs)
                self.update_walk_history(bench_params, history, walk_history)

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
            history = self.fit_model(bench_params, model, callbacks, x_train, y_train, x_test, y_test)
            if bench_params.save_files:
                # restores best epoch of this iteration
                model = load_model(benchmark_file_helper.get_model_path(bench_params))
            accuracy, loss, y_test_prediction = self.evaluate_predict(model, x_test, y_test)
            roc_y_test = y_test

        fpr, tpr, roc_auc = benchmark_roc_auc.calculate_roc_auc(roc_y_test, y_test_prediction,
                                                                bench_params.classes_count)
        return model, accuracy, loss, fpr, tpr, roc_auc, y_test_prediction, history

    def create_model(self, bench_params):
        """Create predicting model"""
        pass

    def create_callbacks(self, bench_params):
        """Create callbacks used while learning"""
        pass

    def evaluate_predict(self, model, x_test, y_test):
        """Evaluate on test data, predict labels for x_test, return (accuracy, loss, y_prediction)"""
        pass

    def fit_model(self, bench_params, model, callbacks, x_train, y_train, x_test, y_test, epochs=None):
        """Fit model on train data, return learning history or none"""
        pass

    def update_walk_history(self, bench_params, history, walk_history):
        """Update history object with walk forward learning history"""
        pass

    def create_history_object(self, bench_params):
        """Create an empty history object for walk forward learning"""
        pass


class NnBenchmark(Benchmark):
    def __init__(self, symbols, bench_params: NnBenchmarkParams, changing_params_dict: dict) -> None:
        super().__init__(symbols, bench_params, changing_params_dict)

    def create_callbacks(self, bench_params):
        earlyStopping = EarlyStopping(monitor='val_' + bench_params.metric,
                                      min_delta=bench_params.early_stopping_min_delta,
                                      patience=bench_params.early_stopping_patience, verbose=0, mode='max',
                                      restore_best_weights=True)
        callbacks = [earlyStopping]
        if bench_params.save_files:
            mcp_save = ModelCheckpoint(
                benchmark_file_helper.get_model_path(bench_params), save_best_only=True,
                monitor='val_' + bench_params.metric, mode='max')
            callbacks = [earlyStopping, mcp_save]
        return callbacks

    def create_model(self, bench_params):
        return benchmark_nn_model.create_seq_model(bench_params)

    def evaluate_predict(self, model, x_test, y_test):
        ls, acc = model.evaluate(x_test, y_test, verbose=0)
        y_test_prediction = model.predict(x_test)
        return acc, ls, y_test_prediction

    def fit_model(self, bench_params, model, callbacks, x_train, y_train, x_test, y_test, epochs=None):
        if epochs is None:
            epochs = bench_params.epochs
        return model.fit(x_train, y_train, validation_data=(x_test, y_test),
                         epochs=epochs, batch_size=bench_params.batch_size,
                         callbacks=callbacks, verbose=0)

    def update_walk_history(self, bench_params, history, walk_history):
        walk_history.history['loss'] += history.history['loss']
        walk_history.history['val_loss'] += history.history['val_loss']
        walk_history.history[bench_params.metric] += history.history[bench_params.metric]
        walk_history.history['val_' + bench_params.metric] += history.history[
            'val_' + bench_params.metric]

    def create_history_object(self, bench_params):
        walk_history = keras.callbacks.History()
        walk_history.history = {'loss': [], 'val_loss': [], bench_params.metric: [],
                                'val_' + bench_params.metric: []}
        return walk_history


if __name__ == '__main__':
    bench_params = benchmark_params.NnBenchmarkParams(False, examined_param='pca,walk_forward_test_window_size',
                                                      benchmark_name='bench-learn-from-scratch')
    bench_params.plot_partial = True
    bench_params.walk_forward_testing = True
    bench_params.walk_forward_learn_from_scratch = True
    bench_params.epochs = 50
    bench_params.iterations = 2
    bench_params.walk_forward_retrain_epochs = 5
    NnBenchmark(['GOOGL'], bench_params, {'pca': [None, 0.9999], 'walk_forward_test_window_size': [300, 200]})
