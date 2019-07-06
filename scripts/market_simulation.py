import time

import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

import benchmark_data_preprocessing
import benchmark_file_helper
import benchmark_nn_model
import csv_importer
import stock_constants
from benchmark_params import BenchmarkParams, NnBenchmarkParams

CSV_TICKER = 'ticker'
CSV_BALANCE = 'balance'
CSV_BUDGET = 'budget'
CSV_BUY_AND_HOLD_BALANCE = 'buy_and_hold_balance'
CSV_SELL_COL = 'sell'
CSV_BUY_COL = 'buy'
CSV_DATE_COL = 'date'

SYMBOLS = ['GOOGL', 'MSFT', 'AAPL', 'CSCO', 'INTC', 'FB', 'PEP', 'QCOM', 'AMZN', 'AMGN']

TRANSACTION_PERCENT_FEE = 0.002


class MarketSimulation:
    def __init__(self, symbols, benchmark_params: BenchmarkParams, date_simulation_start='2019-01-01',
                 budget=100000, verbose=True) -> None:
        benchmark_file_helper.initialize_dirs(bench_params)
        self.symbols = symbols
        self.bench_params = benchmark_params
        self.date_simulation_start = date_simulation_start
        self.budget = budget
        self.current_stock_amount = 0
        self.current_balance = self.budget
        self.buy_and_hold_stock_amount = 0
        self.buy_and_hold_balance = self.budget
        self.verbose = verbose
        self.results_df = pd.DataFrame()
        self.details_results_df = pd.DataFrame()

        benchmark_time = time.time()
        self.df_list, self.sym_list = csv_importer.import_data_from_files(symbols, bench_params.csv_files_path)
        self.run()

        benchmark_file_helper.save_results(self.results_df, bench_params)

        print('Market simulation finished in {0}.'.format(round(time.time() - benchmark_time, 2)))

    def run(self):
        for symbol_it in range(0, len(self.symbols)):
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
            test_df = df[(df.index >= self.date_simulation_start)].copy()
            test_df_len = len(test_df)
            test_size = test_df_len / len(df)
            self.bench_params.test_size = test_size
            x, y, x_train, x_test, y_train, y_test = benchmark_data_preprocessing.preprocess(df, self.bench_params)
            if test_df_len != len(y_test):
                print(
                    'Unexpected test set length for company {0}, expected {1} actually {2}'.format(bench_params.curr_sym
                                                                                                   , test_df_len,
                                                                                                   len(y_test)))
                continue
            if self.bench_params.walk_forward_testing:
                self.bench_params.input_size = x_train[0].shape[1]
            else:
                self.bench_params.input_size = x_train.shape[1]

            self.run_single_company(x_train, y_train, test_df, x_test, y_test)
            benchmark_file_helper.save_results(self.details_results_df, bench_params, bench_params.curr_sym)

            result_dict = {CSV_TICKER: self.bench_params.curr_sym, CSV_BUDGET: self.budget,
                           CSV_BALANCE: self.current_balance, CSV_BUY_AND_HOLD_BALANCE: self.buy_and_hold_balance}
            self.results_df = self.results_df.append(result_dict, ignore_index=True)

    def run_single_company(self, x_train, y_train, test_df, x_test, y_test):
        model = self.create_and_train_model(x_train, y_train, x_test, y_test)
        accuracies = []
        for day in range(0, len(test_df) - 1):
            x_day = np.array([x_test[day, :], ])
            y_day = np.array([y_test[day], ])
            y_test_prediction = self.predict(model, x_day)
            if self.bench_params.binary_classification:
                y_test_prediction_parsed = np.array(y_test_prediction, copy=True)
                y_test_prediction_parsed[y_test_prediction >= 0.5] = 1
                y_test_prediction_parsed[y_test_prediction < 0.5] = 0
            else:
                y_test_prediction_parsed = np.array(
                    [np.argmax(pred, axis=None, out=None) for pred in y_test_prediction])
                y_day = np.array(
                    [np.argmax(pred, axis=None, out=None) for pred in y_day])

            self.manage_account(day, y_test_prediction_parsed, test_df)

            acc = accuracy_score(y_day, y_test_prediction_parsed)
            accuracies.append(acc)

        if self.verbose:
            print(
                'Achieved {0} accuracy for {1}. Finished with {2} dollars which is a {3}% of the budget.\
                 Buy and hold finished with {4} dollars which is a {5}% of the budget.'.format(
                    round(np.mean(accuracies), 4), bench_params.curr_sym, round(self.current_balance, 2)
                    , round(self.current_balance / self.budget * 100, 2)
                    , round(self.buy_and_hold_balance, 2)
                    , round(self.buy_and_hold_balance / self.budget * 100, 2))
            )
        else:
            print(
                '{0} finished with {1}% of the budget. Buy and hold finished with {2}% of the budget.'.format(
                    bench_params.curr_sym
                    , round(self.current_balance / self.budget * 100, 2)
                    , round(self.buy_and_hold_balance / self.budget * 100, 2))
            )

    def create_and_train_model(self, x_train, y_train, x_test, y_test):
        """Create predicting model, return model"""
        return None

    def predict(self, model, x_day):
        """Predict value for one day"""
        return None

    def manage_account(self, day, y_test_prediction_parsed, test_df):
        curr_date = test_df.index.values[day]
        next_day_values = test_df.iloc[day + 1]
        next_day_open_price = next_day_values[stock_constants.OPEN_COL]
        transaction_performed = False
        if self.bench_params.binary_classification:
            y_predicted_value = int(y_test_prediction_parsed[0][0])
            buy_signal = y_predicted_value == stock_constants.IDLE_VALUE
            sell_signal = y_predicted_value == stock_constants.FALL_VALUE
        else:
            y_predicted_value = int(y_test_prediction_parsed[0])
            buy_signal = y_predicted_value == stock_constants.RISE_VALUE
            sell_signal = y_predicted_value == stock_constants.FALL_VALUE
        should_buy = self.current_stock_amount == 0 and buy_signal
        should_sell = self.current_stock_amount > 0 and sell_signal

        if day == 0:
            price_plus_fee = next_day_open_price + next_day_open_price * TRANSACTION_PERCENT_FEE
            self.buy_and_hold_stock_amount = int(self.buy_and_hold_balance / (price_plus_fee))
            self.buy_and_hold_balance = self.buy_and_hold_balance - self.buy_and_hold_stock_amount * price_plus_fee

        if day == len(test_df) - 2:
            transaction_performed = True
            price_minus_fee = next_day_open_price - next_day_open_price * TRANSACTION_PERCENT_FEE
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
                           CSV_BALANCE: estimated_balance, CSV_BUY_AND_HOLD_BALANCE: estimated_buy_and_hold_balance}
            self.details_results_df = self.details_results_df.append(result_dict, ignore_index=True)

    def sell(self, price):
        price_minus_fee = price - price * TRANSACTION_PERCENT_FEE
        self.current_balance = self.current_balance + self.current_stock_amount * price_minus_fee
        if self.verbose:
            print('Selling {0} securities for {1}$ each resulting in {2} dollars'.format(self.current_stock_amount,
                                                                                         price_minus_fee,
                                                                                         self.current_balance))
        self.current_stock_amount = 0

    def buy(self, price):
        price_plus_fee = price + price * TRANSACTION_PERCENT_FEE
        self.current_stock_amount = int(self.current_balance / (price_plus_fee))
        if self.verbose:
            print('Buying {0} securities using {1} dollars ({2} each)'.format(self.current_stock_amount,
                                                                              self.current_balance,
                                                                              price_plus_fee))
        self.current_balance = self.current_balance - self.current_stock_amount * price_plus_fee


class NnMarketSimulation(MarketSimulation):

    def __init__(self, symbols, benchmark_params: NnBenchmarkParams, date_simulation_start='2019-01-01',
                 budget=100000, verbose=False) -> None: \
            super().__init__(symbols, benchmark_params, date_simulation_start, budget, verbose=verbose)

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
        return model.predict(x_day)


if __name__ == '__main__':
    bench_params = NnBenchmarkParams(True benchmark_name='nn-market-simulation')
    NnMarketSimulation(SYMBOLS, bench_params)
