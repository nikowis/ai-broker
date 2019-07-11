import random
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

import benchmark_data_preprocessing
import benchmark_file_helper
import benchmark_nn_model
import csv_importer
import stock_constants
from benchmark_params import BenchmarkParams, NnBenchmarkParams, LightGBMBenchmarkParams, RandomForestBenchmarkParams, \
    SVMBenchmarkParams

CSV_TICKER = 'ticker'
CSV_BALANCE = 'balance'
CSV_BUDGET = 'budget'
CSV_BUY_AND_HOLD_BALANCE = 'buy_and_hold_balance'
CSV_SELL_COL = 'sell'
CSV_BUY_COL = 'buy'
CSV_DATE_COL = 'date'
CSV_PRICE_COL = 'price'
CSV_TRAIN_TIME_COL = 'train_time'

TRANSACTION_PERCENT_FEE = 0.003
AVERAGE_SPREAD = 0.003


class MarketSimulation:
    def __init__(self, symbols, benchmark_params: BenchmarkParams, date_simulation_start='2019-01-01',
                 date_simulation_end='2099-01-01',
                 budget=100000, verbose=True) -> None:
        self.bench_params = benchmark_params
        print('Begin simulation ', benchmark_params.benchmark_name)
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
        self.details_results_df = pd.DataFrame()

        benchmark_time = time.time()
        self.df_list, self.sym_list = csv_importer.import_data_from_files(symbols, self.bench_params.csv_files_path)
        self.run()

        benchmark_file_helper.save_results(self.results_df, self.bench_params)

        print('Market simulation finished in {0}.'.format(round(time.time() - benchmark_time, 2)))

    def run(self):
        accuracies = []
        balances = []
        for symbol_it in range(0, len(self.symbols)):
            single_company_benchmark_time = time.time()
            self.bench_params.curr_sym = self.symbols[symbol_it]
            self.current_stock_amount = 0
            self.current_balance = self.budget
            self.buy_and_hold_stock_amount = 0
            self.buy_and_hold_balance = self.budget
            self.details_results_df = pd.DataFrame(
                {CSV_DATE_COL: [], CSV_BUY_COL: [], CSV_SELL_COL: [], CSV_BALANCE: [], CSV_BUY_AND_HOLD_BALANCE: []})
            self.details_results_df = self.details_results_df.set_index(CSV_DATE_COL)

            df = self.df_list[symbol_it]
            df.dropna(inplace=True)
            df = df[(df.index <= self.date_simulation_end)]
            test_df = df[(df.index >= self.date_simulation_start)].copy()
            if len(df) < 300:
                print("{0} not enough data to perform simulation. {1} samples found between {2} and {3}.".format(
                    self.bench_params.curr_sym, len(df), self.date_simulation_start, self.date_simulation_end))
                continue
            elif len(df) - len(test_df) < 200:
                print(
                    "{0} training set too small to perform simulation. {1} samples in training set and {2} samples in test set.".format(
                        self.bench_params.curr_sym, len(df) - len(test_df), len(test_df)))
                continue
            test_df_len = len(test_df)
            test_size = test_df_len / len(df)
            self.bench_params.test_size = test_size
            x, y, x_train, x_test, y_train, y_test, _, _ = benchmark_data_preprocessing.preprocess(df,
                                                                                                   self.bench_params)

            if self.bench_params.walk_forward_testing:
                self.bench_params.input_size = x_train[0].shape[1]
            else:
                self.bench_params.input_size = x_train.shape[1]

            acc = self.run_single_company(x_train, y_train, test_df, x_test, y_test)
            accuracies.append(acc)
            balances.append(self.current_balance)
            benchmark_file_helper.save_results(self.details_results_df, self.bench_params, self.bench_params.curr_sym)

            result_dict = {CSV_TICKER: self.bench_params.curr_sym, CSV_BUDGET: self.budget,
                           CSV_BALANCE: self.current_balance, CSV_BUY_AND_HOLD_BALANCE: self.buy_and_hold_balance,
                           CSV_TRAIN_TIME_COL: round(time.time() - single_company_benchmark_time, 2)}
            self.results_df = self.results_df.append(result_dict, ignore_index=True)
        print('Overall accuracy {0} and balance {1}'.format(round(np.mean(accuracies), 4), round(np.mean(balances), 4)))

    def run_single_company(self, x_train, y_train, test_df, x_test, y_test):
        predictions = []
        expected_y = []

        if self.bench_params.walk_forward_testing:
            walk_iterations = len(x_train)
            for walk_it in range(0, walk_iterations):
                walk_x_train = x_train[walk_it]
                walk_x_test = x_test[walk_it]
                walk_y_train = y_train[walk_it]
                walk_y_test = y_test[walk_it]
                offset = walk_it * self.bench_params.walk_forward_test_window_size
                self.simulate_on_data_batch(expected_y, predictions, test_df, walk_x_train, walk_y_train,
                                            walk_x_test, walk_y_test, days_offset=offset)
        else:
            self.simulate_on_data_batch(expected_y, predictions, test_df, x_train, y_train, x_test, y_test)

        acc = accuracy_score(predictions, expected_y)

        if self.verbose:
            print(
                'Achieved {0} accuracy for {1}. Finished with {2} dollars which is a {3}% of the budget.\
                 Buy and hold finished with {4} dollars which is a {5}% of the budget.'.format(
                    round(np.mean(acc), 4), self.bench_params.curr_sym, round(self.current_balance, 2)
                    , round(self.current_balance / self.budget * 100, 2)
                    , round(self.buy_and_hold_balance, 2)
                    , round(self.buy_and_hold_balance / self.budget * 100, 2))
            )
        else:
            print(
                'Achieved {0} accuracy for {1}. Finished with {2}% of the budget. Buy and hold finished with {3}% of the budget.'.format(
                    round(np.mean(acc), 4), self.bench_params.curr_sym
                    , round(self.current_balance / self.budget * 100, 2)
                    , round(self.buy_and_hold_balance / self.budget * 100, 2))
            )

        return acc

    def simulate_on_data_batch(self, expected_y, predictions, test_df, x_train, y_train, x_test, y_test,
                               days_offset=0):
        self.bench_params.input_size = x_train.shape[1]
        model = self.create_and_train_model(x_train, y_train, x_test, y_test)
        if self.bench_params.walk_forward_testing:
            end = days_offset + len(y_test)
        else:
            end = len(test_df) - 1
        for day in range(days_offset, end):
            day_index = day - days_offset
            x_day = np.array([x_test[day_index, :], ])
            if self.bench_params.binary_classification or not self.bench_params.one_hot_encode_labels:
                y_day = y_test[day_index]
            else:
                y_day = np.argmax(y_test[day_index], axis=None, out=None)
            y_test_prediction = self.predict(model, x_day)
            predictions.append(y_test_prediction)
            expected_y.append(y_day)
            self.manage_account(day, y_test_prediction, test_df)

    def create_and_train_model(self, x_train, y_train, x_test, y_test):
        """Create predicting model, return model"""
        return None

    def predict(self, model, x_day):
        """Predict value for one day"""
        return None

    def manage_account(self, day, y_predicted_value, test_df):
        curr_date = test_df.index.values[day]
        if day + 1 >= len(test_df):
            return
        next_day_values = test_df.iloc[day + 1]
        next_day_open_price = next_day_values[stock_constants.OPEN_COL]
        transaction_performed = False
        if self.bench_params.binary_classification:
            buy_signal = y_predicted_value == stock_constants.IDLE_VALUE
            sell_signal = y_predicted_value == stock_constants.FALL_VALUE
        else:
            buy_signal = y_predicted_value == stock_constants.RISE_VALUE
            sell_signal = y_predicted_value == stock_constants.FALL_VALUE
        should_buy = self.current_stock_amount == 0 and buy_signal
        should_sell = self.current_stock_amount > 0 and sell_signal

        if day == 0:
            price_plus_fee = next_day_open_price + next_day_open_price * (TRANSACTION_PERCENT_FEE + AVERAGE_SPREAD)
            self.buy_and_hold_stock_amount = int(self.buy_and_hold_balance / (price_plus_fee))
            self.buy_and_hold_balance = self.buy_and_hold_balance - self.buy_and_hold_stock_amount * price_plus_fee

        if day == len(test_df) - 2:
            transaction_performed = True
            price_minus_fee = next_day_open_price - next_day_open_price * (TRANSACTION_PERCENT_FEE + AVERAGE_SPREAD)
            self.buy_and_hold_balance = self.buy_and_hold_balance + self.buy_and_hold_stock_amount * price_minus_fee
            self.buy_and_hold_stock_amount = 0
            if self.current_stock_amount > 0:
                self.sell(next_day_open_price)
            if self.verbose:
                print('Selling all stock at the end of learning')
        elif should_buy:
            transaction_performed = True
            self.buy(next_day_open_price)
        elif should_sell:
            transaction_performed = True
            self.sell(next_day_open_price)

        if transaction_performed:
            if self.current_stock_amount > 0:
                estimated_balance = self.current_balance + next_day_open_price * self.current_stock_amount
            else:
                estimated_balance = self.current_balance
            estimated_buy_and_hold_balance = self.buy_and_hold_balance + next_day_open_price * self.buy_and_hold_stock_amount

            result_dict = {CSV_DATE_COL: curr_date, CSV_BUY_COL: should_buy, CSV_SELL_COL: should_sell,
                           CSV_BALANCE: estimated_balance, CSV_BUY_AND_HOLD_BALANCE: estimated_buy_and_hold_balance,
                           CSV_PRICE_COL: next_day_open_price}
            self.details_results_df = self.details_results_df.append(result_dict, ignore_index=True)

    def sell(self, price):
        price_minus_fee = price - price * (TRANSACTION_PERCENT_FEE + AVERAGE_SPREAD)
        self.current_balance = self.current_balance + self.current_stock_amount * price_minus_fee
        if self.verbose:
            print('Selling {0} securities for {1}$ each resulting in {2} dollars'.format(self.current_stock_amount,
                                                                                         price_minus_fee,
                                                                                         self.current_balance))
        self.current_stock_amount = 0

    def buy(self, price):
        price_plus_fee = price + price * (TRANSACTION_PERCENT_FEE + AVERAGE_SPREAD)
        self.current_stock_amount = int(self.current_balance / (price_plus_fee))
        if self.verbose:
            print('Buying {0} securities using {1} dollars ({2} each)'.format(self.current_stock_amount,
                                                                              self.current_balance,
                                                                              price_plus_fee))
        self.current_balance = self.current_balance - self.current_stock_amount * price_plus_fee


class NnMarketSimulation(MarketSimulation):

    def __init__(self, symbols, benchmark_params: NnBenchmarkParams, date_simulation_start='2019-01-01',
                 date_simulation_end='2099-01-01',
                 budget=100000, verbose=False) -> None: \
            super().__init__(symbols, benchmark_params, date_simulation_start, date_simulation_end, budget,
                             verbose=verbose)

    def create_and_train_model(self, x_train, y_train, x_test, y_test):
        bench_params = self.bench_params
        model = benchmark_nn_model.create_seq_model(bench_params)

        earlyStopping = EarlyStopping(monitor='val_' + bench_params.metric
                                      , min_delta=bench_params.early_stopping_min_delta
                                      , patience=bench_params.early_stopping_patience
                                      , verbose=0, mode='max'
                                      , restore_best_weights=True)

        model.fit(x_train, y_train, validation_data=(x_test, y_test),
                  epochs=bench_params.epochs, batch_size=bench_params.batch_size,
                  callbacks=[earlyStopping], verbose=0)
        return model

    def predict(self, model, x_day):
        y_test_prediction = model.predict(x_day)
        if self.bench_params.binary_classification:
            y_test_prediction_parsed = np.array(y_test_prediction, copy=True)
            y_test_prediction_parsed[y_test_prediction >= 0.5] = 1
            y_test_prediction_parsed[y_test_prediction < 0.5] = 0
        else:
            y_test_prediction_parsed = np.array(
                [np.argmax(pred, axis=None, out=None) for pred in y_test_prediction])
        return int(y_test_prediction_parsed[0])


class LightGBMSimulation(MarketSimulation):

    def __init__(self, symbols, benchmark_params: LightGBMBenchmarkParams, date_simulation_start='2019-01-01',
                 date_simulation_end='2099-01-01',
                 budget=100000, verbose=False) -> None: \
            super().__init__(symbols, benchmark_params, date_simulation_start, date_simulation_end, budget,
                             verbose=verbose)

    def create_and_train_model(self, x_train, y_train, x_test, y_test):
        bench_params = self.bench_params
        params = {
            "objective": bench_params.objective,
            "num_class": bench_params.model_num_class,
            "num_leaves": bench_params.num_leaves,
            "max_depth": bench_params.max_depth,
            "learning_rate": bench_params.learning_rate,
            "boosting": bench_params.boosting,
            "num_threads": 2,
            "max_bin": bench_params.max_bin,
            # "bagging_fraction" : bench_params.bagging_fraction,
            # "bagging_freq" : bench_params.bagging_freq,
            "feature_fraction": bench_params.feature_fraction,
            "min_sum_hessian_in_leaf": bench_params.min_sum_hessian_in_leaf,
            "min_data_in_leaf": bench_params.min_data_in_leaf,
            "verbosity": -1
        }

        train_data = lgb.Dataset(x_train, label=y_train)
        test_data = lgb.Dataset(x_test, label=y_test)
        bst = lgb.train(params, train_data, valid_sets=[test_data, train_data],
                        num_boost_round=bench_params.num_boost_round
                        , early_stopping_rounds=20, verbose_eval=False
                        , feature_name=bench_params.feature_names)
        return bst

    def predict(self, model, x_day):
        y_test_prediction = model.predict(x_day, num_iteration=model.best_iteration)
        if self.bench_params.binary_classification:
            y_test_prediction_parsed = np.array(y_test_prediction, copy=True)
            y_test_prediction_parsed[y_test_prediction >= 0.5] = 1
            y_test_prediction_parsed[y_test_prediction < 0.5] = 0
        else:
            y_test_prediction_parsed = np.array([np.argmax(pred, axis=None, out=None) for pred in y_test_prediction])
        return int(y_test_prediction_parsed[0])


class RandomForestSimulation(MarketSimulation):

    def __init__(self, symbols, benchmark_params: RandomForestBenchmarkParams, date_simulation_start='2019-01-01',
                 date_simulation_end='2099-01-01',
                 budget=100000, verbose=False) -> None: \
            super().__init__(symbols, benchmark_params, date_simulation_start, date_simulation_end, budget,
                             verbose=verbose)

    def create_and_train_model(self, x_train, y_train, x_test, y_test):
        bench_params = self.bench_params
        model = RandomForestClassifier(n_estimators=bench_params.n_estimators, criterion=bench_params.criterion
                                       , max_depth=bench_params.max_depth,
                                       min_samples_split=bench_params.min_samples_split
                                       , min_samples_leaf=bench_params.min_samples_leaf
                                       , min_weight_fraction_leaf=bench_params.min_weight_fraction_leaf
                                       , max_features=bench_params.max_features
                                       , max_leaf_nodes=bench_params.max_leaf_nodes
                                       , min_impurity_decrease=bench_params.min_impurity_decrease
                                       , bootstrap=bench_params.bootstrap
                                       , warm_start=bench_params.warm_start
                                       , oob_score=bench_params.oob_score)
        model.fit(x_train, y_train)
        return model

    def predict(self, model, x_day):
        value = model.predict(x_day)
        return int(value[0])


class SVMSimulation(MarketSimulation):

    def __init__(self, symbols, benchmark_params: SVMBenchmarkParams, date_simulation_start='2019-01-01',
                 date_simulation_end='2099-01-01',
                 budget=100000, verbose=False) -> None: \
            super().__init__(symbols, benchmark_params, date_simulation_start, date_simulation_end, budget,
                             verbose=verbose)

    def create_and_train_model(self, x_train, y_train, x_test, y_test):
        bench_params = self.bench_params
        model = SVC(C=bench_params.c, kernel=bench_params.kernel, degree=bench_params.degree, gamma=bench_params.gamma,
                    probability=True)
        model.fit(x_train, y_train)
        return model

    def predict(self, model, x_day):
        value = model.predict(x_day)
        return int(value[0])


class RandomSimulation(MarketSimulation):

    def __init__(self, symbols, benchmark_params: BenchmarkParams, date_simulation_start='2019-01-01',
                 budget=100000, verbose=False) -> None: \
            super().__init__(symbols, benchmark_params, date_simulation_start, budget, verbose=verbose)

    def create_and_train_model(self, x_train, y_train, x_test, y_test):
        return None

    def predict(self, model, x_day):
        if self.bench_params.binary_classification:
            value = random.randint(0, 1)
        else:
            value = random.randint(0, 2)
        return value


if __name__ == '__main__':
    bench_params = NnBenchmarkParams(False, benchmark_name='nn-market-simulation')
    bench_params.walk_forward_testing = True
    NnMarketSimulation(['AAL', 'CTRP', 'INTC'], bench_params,
                       date_simulation_start='2018-01-01', date_simulation_end='2019-07-01')
    bench_params = SVMBenchmarkParams(False, benchmark_name='svm-market-simulation')
    SVMSimulation(['AAL', 'CTRP', 'INTC'], bench_params,
                  date_simulation_start='2018-01-01', date_simulation_end='2019-07-01')
    print('Finished all')
