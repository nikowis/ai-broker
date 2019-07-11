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
from data_import.api_to_db_importer import Importer

SYMBOLS = ['AAL']
TARGET_PATH = './../target'
CSV_FILES_PATH = TARGET_PATH + '/data'


def create_svm(x_train, y_train, bench_params):
    svm = SVC(C=bench_params.c, kernel=bench_params.kernel, degree=bench_params.degree,
              gamma=bench_params.gamma,
              probability=True)
    svm.fit(x_train, y_train)
    acc = svm.score(x_train, y_train)
    print("Finished training model for {0} train accuracy {1}".format(bench_params.curr_sym, acc))
    return svm

if __name__ == '__main__':
    imp = Importer()
    # imp.import_all(SYMBOLS, False)
    # imp.import_all_technical_indicators(SYMBOLS)
    # imp.process_data(True)
    # imp.export_to_csv_files('./../target/data')
    bench_params = SVMBenchmarkParams(False)
    bench_params.walk_forward_testing = False
    df_list, sym_list = csv_importer.import_data_from_files(SYMBOLS, CSV_FILES_PATH)

    for symbol_it in range(0, len(SYMBOLS)):
        df = df_list[symbol_it]
        sym = sym_list[symbol_it]
        bench_params.curr_sym = sym
        bench_params.test_size = 0
        last_idx = df.index.values[len(df)-1]
        close_value = df.iloc[len(df)-1][stock_constants.ADJUSTED_CLOSE_COL]
        print('{0} last index {1} close value {2}'.format(sym, last_idx, close_value))
        train_df = df[(df.index < last_idx)]
        test_processed_df = benchmark_data_preprocessing.manage_and_drop_helper_df_columns(df.copy(),
                                                                                         bench_params.difference_non_stationary)
        test_df = test_processed_df[(test_processed_df.index >= last_idx)]
        x, y, x_train, _, y_train, _, std_scaler, pca_transformer = benchmark_data_preprocessing.preprocess(train_df, bench_params)
        bench_params.input_size = x_train.shape[1]
        if test_df.isnull().any(axis=1).iloc[0]:
            print("{0} CRITICAL ERR FOUND NULLS IN ROW".format(sym))
            continue
        x_today = np.array(test_df.iloc[0])
        x_today = x_today.reshape(1, -1)
        if std_scaler is not None:
            x_today = std_scaler.transform(x_today)
        if pca_transformer is not None:
            x_today = pca_transformer.transform(x_today)

        svm = create_svm(x_train, y_train, bench_params)

        prediction = svm.predict(x_today)[0]
        if prediction == 0:
            prediction_str = 'sell'
        elif prediction == 2:
            prediction_str = 'buy'
        else:
            prediction_str = 'hold'
        print("{0} prediction from {1} is {2} == {3} tommorow".format(sym, last_idx, prediction, prediction_str))
    print('Finished predicting')

