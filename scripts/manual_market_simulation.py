import time

import numpy as np
import pandas as pd
from sklearn.svm import SVC

import benchmark_data_preprocessing
import benchmark_file_helper
import csv_importer
import stock_constants
from benchmark_params import BenchmarkParams, SVMBenchmarkParams

CSV_TICKER = 'ticker'
CSV_DATE_COL = 'date'
CSV_TODAY_OPEN_COL = 'today_open'
CSV_TODAY_CLOSE_COL = 'today_close'
CSV_PREDICTION_COL = 'tommorow_prediction'


class ManualMarketSimulation:
    def __init__(self, symbols, benchmark_params: BenchmarkParams, date_simulation_start='2019-01-01',
                 date_simulation_end='2099-01-01',
                 budget=100000, verbose=True) -> None:
        self.bench_params = benchmark_params
        print('Begin manual simulation', benchmark_params.benchmark_name)
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
                                                                                     bench_params.difference_non_stationary)
            test_reduced_df = test_reduced_df.iloc[1:]
            test_df = test_df.iloc[1:]
            self.bench_params.test_size = 0
            x, y, x_train, _, y_train, _, std_scaler, pca_transformer = benchmark_data_preprocessing.preprocess(
                train_df, self.bench_params)

            if self.bench_params.walk_forward_testing:
                self.bench_params.input_size = x_train[0].shape[1]
            else:
                self.bench_params.input_size = x_train.shape[1]

            model = self.create_and_train_model(x_train, y_train)
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
                y_test_prediction = self.predict(model, x_day)
                print("Predicting {0} sample {1}. Prediction: {2}".format(self.bench_params.curr_sym, day_date,
                                                                          y_test_prediction))
                result_dict = {CSV_TICKER: self.bench_params.curr_sym, CSV_DATE_COL: day_date,
                               CSV_TODAY_OPEN_COL: open_col, CSV_TODAY_CLOSE_COL: close_col,
                               CSV_PREDICTION_COL: y_test_prediction}
                self.results_df = self.results_df.append(result_dict, ignore_index=True)

    def create_and_train_model(self, x_train, y_train):
        """Create predicting model, return model"""
        return None

    def predict(self, model, x_day):
        """Predict value for one day"""
        return None


class SVMManualSimulation(ManualMarketSimulation):

    def __init__(self, symbols, benchmark_params: SVMBenchmarkParams, date_simulation_start='2019-03-31',
                 date_simulation_end='2019-06-1',
                 budget=100000, verbose=False) -> None: \
            super().__init__(symbols, benchmark_params, date_simulation_start, date_simulation_end, budget,
                             verbose=verbose)

    def create_and_train_model(self, x_train, y_train):
        bench_params = self.bench_params
        model = SVC(C=bench_params.c, kernel=bench_params.kernel, degree=bench_params.degree, gamma=bench_params.gamma,
                    probability=True)
        model.fit(x_train, y_train)
        acc = model.score(x_train, y_train)
        print("Finished training model for {0} train accuracy {1}".format(self.bench_params.curr_sym, acc))
        return model

    def predict(self, model, x_day):
        value = model.predict(x_day)
        return int(value[0])


if __name__ == '__main__':
    bench_params = SVMBenchmarkParams(False, benchmark_name='svm-manual-market-simulation')
    SVMManualSimulation(['GOOGL', 'INTC', 'MSFT'], bench_params)

    print('Finished all')
