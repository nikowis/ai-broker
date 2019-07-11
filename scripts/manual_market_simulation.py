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
CSV_SVM_PREDICTION_COL = 'svm_tommorow_prediction'
CSV_LGBM_PREDICTION_COL = 'lgbm_tommorow_prediction'
CSV_NN_PREDICTION_COL = 'nn_tommorow_prediction'
RESULT_PATH = './../../target/results/'

BUDGET = 100000
FEE = 0.004

class ManualMarketSimulation:
    def __init__(self, symbols, binary_classification, simulation_name, date_simulation_start='2019-01-01',
                 date_simulation_end='2099-01-01',
                 budget=100000, verbose=False) -> None:
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

        print('Market manual simulation finished in {0}.'.format(round(time.time() - benchmark_time, 2)))


    def run(self):
        self.bench_params.walk_forward_testing = False
        for symbol_it in range(0, len(self.symbols)):
            self.bench_params.curr_sym = self.symbols[symbol_it]
            self.current_stock_amount = 0
            self.current_balance = self.budget
            self.buy_and_hold_stock_amount = 0
            self.buy_and_hold_balance = self.budget
            self.results_df = pd.DataFrame()

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

            self.create_svm(x_train, y_train)
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

                svm_prediction, lgbm_prediction, nn_prediction = self.predict(x_day)
                if self.verbose:
                    print("Predicting {0} sample {1}. Prediction: {2} {3} {4}".format(self.bench_params.curr_sym,
                                                                                      day_date,
                                                                                      svm_prediction, lgbm_prediction,
                                                                                      nn_prediction))
                result_dict = {CSV_TICKER: self.bench_params.curr_sym, CSV_DATE_COL: day_date,
                               CSV_TODAY_OPEN_COL: open_col, CSV_TODAY_CLOSE_COL: close_col,
                               CSV_NN_PREDICTION_COL: nn_prediction, CSV_SVM_PREDICTION_COL: svm_prediction,
                               CSV_LGBM_PREDICTION_COL: lgbm_prediction}
                self.results_df = self.results_df.append(result_dict, ignore_index=True)
            benchmark_file_helper.save_results(self.results_df, self.bench_params, self.bench_params.curr_sym)
            self.results_df.set_index('date', inplace=True)
            self.results_df.index = pd.to_datetime(self.results_df.index)
            self.analyze_results(CSV_SVM_PREDICTION_COL)
            self.analyze_results(CSV_LGBM_PREDICTION_COL)
            self.analyze_results(CSV_NN_PREDICTION_COL)

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
                        , early_stopping_rounds=20, verbose_eval=False)
        self.lgbm = bst

    def create_nn(self, x_train, y_train):
        nn_bench_params = NnBenchmarkParams(self.bench_params.binary_classification)
        nn_bench_params.input_size = self.bench_params.input_size
        nn = benchmark_nn_model.create_seq_model(nn_bench_params)

        earlyStopping = EarlyStopping(monitor=nn_bench_params.metric
                                      , min_delta=nn_bench_params.early_stopping_min_delta
                                      , patience=nn_bench_params.early_stopping_patience
                                      , verbose=0, mode='max'
                                      , restore_best_weights=True)

        nn.fit(x_train, y_train,
               epochs=nn_bench_params.epochs, batch_size=nn_bench_params.batch_size,
               callbacks=[earlyStopping], verbose=0)
        self.nn = nn

    def predict(self, x_day):
        value_svm = self.svm.predict(x_day)[0]
        value_lgbm = [np.argmax(pred, axis=None, out=None) for pred in self.lgbm.predict(x_day)][0]
        value_nn = [np.argmax(pred, axis=None, out=None) for pred in self.nn.predict(x_day)][0]
        return value_svm, value_lgbm, value_nn

    def analyze_results(self, prediction_column):
        cur_money = BUDGET
        buy_and_hold_money = BUDGET
        cur_securities = 0
        buy_and_hold_securities = 0

        df = self.results_df

        company = df.iloc[0][CSV_TICKER]

        today_action = 1

        for day_ix in range(0, len(df)):

            day_date = pd.to_datetime(df.index.values[day_ix])
            today = df.iloc[day_ix]
            prediction = today[prediction_column]

            today_buy_price = today[CSV_TODAY_OPEN_COL] * (1 + FEE)
            today_sell_price = today[CSV_TODAY_OPEN_COL] * (1 - FEE)

            if day_ix == 0:
                buy_and_hold_securities = int(buy_and_hold_money / today_buy_price)
                buy_and_hold_money = buy_and_hold_money - buy_and_hold_securities * today_buy_price

            if today_action == 0 and cur_securities != 0:
                cur_money = cur_money + cur_securities * today_sell_price
                cur_securities = 0
            elif today_action == 2 and cur_securities == 0:
                cur_securities = int(cur_money / today_buy_price)
                cur_money = cur_money - cur_securities * today_buy_price

            if prediction == 0:
                prediction_str = 'sell'
            elif prediction == 2:
                prediction_str = 'buy'
            else:
                prediction_str = 'hold'
            if self.verbose:
                print(
                    '{0} wallet {1} securities and {2} dollars action for tommorow {3}. Buy and hold worth {4}.'.format(
                        day_date.date(), cur_securities,
                        round(cur_money, 2),
                        prediction_str, round(buy_and_hold_money + buy_and_hold_securities * today_sell_price, 2)))

            today_action = prediction

        buy_and_hold_money = buy_and_hold_money + buy_and_hold_securities * today_sell_price
        if cur_securities > 0:
            cur_money = cur_money + cur_securities * today_sell_price

        print('Prediction col {0}. {1} finished with {2}% of the budget. Buy and hold finished with {3}% of the budget.'.format(prediction_column, company,
                                                                                                              round(
                                                                                                                  cur_money / BUDGET * 100,
                                                                                                                  2),
                                                                                                              round(
                                                                                                                  buy_and_hold_money / BUDGET * 100,
                                                                                                                  2)))

if __name__ == '__main__':
    ManualMarketSimulation(stock_constants.BASE_COMPANIES+stock_constants.CHEAP_COMPANIES, False, 'manual--market-simulation',
                           date_simulation_start='2019-01-01', date_simulation_end='2019-07-01')
    print('Finished all')
