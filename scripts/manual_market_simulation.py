import time

import lightgbm as lgb

import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from sklearn.svm import SVC

import benchmark_data_preprocessing
import benchmark_file_helper
import benchmark_nn_model
import csv_importer
import stock_constants
from benchmark_params import BenchmarkParams, SVMBenchmarkParams, NnBenchmarkParams, LightGBMBenchmarkParams

CSV_TICKER = 'ticker'
CSV_DATE_COL = 'date'
CSV_TODAY_OPEN_COL = 'today_open'
CSV_TODAY_CLOSE_COL = 'today_close'
CSV_PREDICTION_COL = 'tommorow_prediction'


class ManualMarketSimulation:
    def __init__(self, symbols, binary_classification, simulation_name, date_simulation_start='2019-01-01',
                 date_simulation_end='2099-01-01',
                 budget=100000, verbose=True) -> None:
        self.bench_params = BenchmarkParams(binary_classification, benchmark_name=simulation_name)
        print('Begin manual simulation', self.bench_params.benchmark_name)
        benchmark_file_helper.initialize_dirs(self.bench_params)
        self.symbols = symbols
        self.date_simulation_start = date_simulation_start
        self.date_simulation_end = date_simulation_end
        self.budget = budget
        self.current_stock_amount = 0
        self.current_balance = self.budget
        self.buy_and_hold_stock_amount = 0
        self.buy_and_hold_balance = self.budget
        self.verbose = verbose
        self.results_df = pd.DataFrame()
        self.svm = None
        self.nn = None
        self.lgbm = None
        benchmark_time = time.time()
        self.df_list, self.sym_list = csv_importer.import_data_from_files(symbols, self.bench_params.csv_files_path)
        self.run()

        benchmark_file_helper.save_results(self.results_df, self.bench_params)

        print('Market manual simulation finished in {0}.'.format(round(time.time() - benchmark_time, 2)))

    def run(self):
        self.bench_params.walk_forward_testing = False
        for symbol_it in range(0, len(self.symbols)):
            self.bench_params.curr_sym = self.symbols[symbol_it]
            self.current_stock_amount = 0
            self.current_balance = self.budget
            self.buy_and_hold_stock_amount = 0
            self.buy_and_hold_balance = self.budget

            df = self.df_list[symbol_it]
            df.dropna(inplace=True)
            df = df[(df.index <= self.date_simulation_end)]
            train_df = df[(df.index <= self.date_simulation_start)].copy()
            test_df = df[(df.index >= self.date_simulation_start)].copy()
            test_reduced_df = benchmark_data_preprocessing.manage_and_drop_helper_df_columns(test_df.copy(),
                                                                                             self.bench_params.difference_non_stationary)
            test_reduced_df = test_reduced_df.iloc[1:]
            test_df = test_df.iloc[1:]
            self.bench_params.test_size = 0
            x, y, x_train, _, y_train, _, std_scaler, pca_transformer = benchmark_data_preprocessing.preprocess(
                train_df, self.bench_params)

            if self.bench_params.walk_forward_testing:
                self.bench_params.input_size = x_train[0].shape[1]
            else:
                self.bench_params.input_size = x_train.shape[1]

            # self.create_svm(x_train, y_train)
            self.create_lgbm(x_train, y_train)
            self.create_nn(x_train, y_train)

            for day_ix in range(0, len(test_reduced_df)):
                unnormalized_day_df = test_df.iloc[day_ix]
                day_date = test_df.index.values[day_ix]

                open_col = unnormalized_day_df[stock_constants.OPEN_COL]
                close_col = unnormalized_day_df[stock_constants.ADJUSTED_CLOSE_COL]
                x_day = np.array(test_reduced_df.iloc[day_ix])
                x_day = x_day.reshape(1, -1)
                if std_scaler is not None:
                    x_day = std_scaler.transform(x_day)
                if pca_transformer is not None:
                    x_day = pca_transformer.transform(x_day)
                y_test_prediction = self.predict(x_day)
                print("Predicting {0} sample {1}. Prediction: {2}".format(self.bench_params.curr_sym, day_date,
                                                                          y_test_prediction))
                result_dict = {CSV_TICKER: self.bench_params.curr_sym, CSV_DATE_COL: day_date,
                               CSV_TODAY_OPEN_COL: open_col, CSV_TODAY_CLOSE_COL: close_col,
                               CSV_PREDICTION_COL: y_test_prediction}
                self.results_df = self.results_df.append(result_dict, ignore_index=True)

    def create_svm(self, x_train, y_train):
        svm_bench_params = SVMBenchmarkParams(self.bench_params.binary_classification)
        svm_bench_params.input_size = self.bench_params.input_size
        svm = SVC(C=svm_bench_params.c, kernel=svm_bench_params.kernel, degree=svm_bench_params.degree,
                  gamma=svm_bench_params.gamma,
                  probability=True)
        y_train = np.array([np.argmax(pred, axis=None, out=None) for pred in y_train])
        svm.fit(x_train, y_train)
        acc = svm.score(x_train, y_train)
        print("Finished training model for {0} train accuracy {1}".format(self.bench_params.curr_sym, acc))
        self.svm = svm

    def create_lgbm(self, x_train, y_train):
        lgbm_bench_params = LightGBMBenchmarkParams(self.bench_params.binary_classification)
        lgbm_bench_params.input_size = self.bench_params.input_size
        params = {
            "objective": lgbm_bench_params.objective,
            "num_class": lgbm_bench_params.model_num_class,
            "num_leaves": lgbm_bench_params.num_leaves,
            "max_depth": lgbm_bench_params.max_depth,
            "learning_rate": lgbm_bench_params.learning_rate,
            "boosting": lgbm_bench_params.boosting,
            "num_threads": 2,
            "max_bin": lgbm_bench_params.max_bin,
            "feature_fraction": lgbm_bench_params.feature_fraction,
            "min_sum_hessian_in_leaf": lgbm_bench_params.min_sum_hessian_in_leaf,
            "min_data_in_leaf": lgbm_bench_params.min_data_in_leaf,
            "verbosity": -1
        }
        y_train = np.array([np.argmax(pred, axis=None, out=None) for pred in y_train])
        train_data = lgb.Dataset(x_train, label=y_train)
        bst = lgb.train(params, train_data,
                        num_boost_round=lgbm_bench_params.num_boost_round
                        , early_stopping_rounds=20, verbose_eval=20)
        self.lgbm = bst

    def create_nn(self, x_train, y_train):
        nn_bench_params = NnBenchmarkParams(self.bench_params.binary_classification)
        nn_bench_params.input_size = self.bench_params.input_size
        nn = benchmark_nn_model.create_seq_model(nn_bench_params)

        earlyStopping = EarlyStopping(monitor= nn_bench_params.metric
                                      , min_delta=nn_bench_params.early_stopping_min_delta
                                      , patience=nn_bench_params.early_stopping_patience
                                      , verbose=0, mode='max'
                                      , restore_best_weights=True)

        nn.fit(x_train, y_train,
               epochs=nn_bench_params.epochs, batch_size=nn_bench_params.batch_size,
               callbacks=[earlyStopping], verbose=0)
        self.nn = nn

    def predict(self, x_day):
        value_svm = self.svm.predict(x_day)
        value_lgbm = self.lgbm.predict(x_day)
        value_nn = self.nn.predict(x_day)
        return int(value_svm[0])


if __name__ == '__main__':
    ManualMarketSimulation(stock_constants.BASE_COMPANIES, False, 'manual-market-simulation',
                           date_simulation_start='2019-03-30', date_simulation_end='2019-06-01')
    print('Finished all')
