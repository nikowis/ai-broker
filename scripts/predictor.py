import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping

import benchmark_data_preprocessing
import benchmark_nn_model
import csv_importer
import stock_constants
from benchmark_params import NnBenchmarkParams, BenchmarkParams
from data_import.api_to_db_importer import Importer

SYMBOLS = stock_constants.BASE_COMPANIES
TARGET_PATH = './../target'
CSV_FILES_PATH = TARGET_PATH + '/data'


def create_nn(x_train, y_train, params):
    model = benchmark_nn_model.create_seq_model(params)

    earlyStopping = EarlyStopping(monitor=params.metric
                                  , min_delta=params.early_stopping_min_delta
                                  , patience=params.early_stopping_patience
                                  , verbose=1, mode='max'
                                  , restore_best_weights=True)

    model.fit(x_train, y_train, epochs=params.epochs, batch_size=params.batch_size,
              callbacks=[earlyStopping], verbose=0)

    acc = model.evaluate(x_train, y_train)[1]

    print("Finished training model for {0} train accuracy {1}".format(params.curr_sym, acc))
    return model


def teach_and_predict(params: NnBenchmarkParams, train_df, test_df):
    x, y, x_train, _, y_train, _, std_scaler, pca_transformer = benchmark_data_preprocessing.preprocess(train_df,
                                                                                                        params)
    params.input_size = x_train.shape[1]

    x_today = np.array(test_df.iloc[0])
    x_today = x_today.reshape(1, -1)
    if std_scaler is not None:
        x_today = std_scaler.transform(x_today)
    if pca_transformer is not None:
        x_today = pca_transformer.transform(x_today)

    model = create_nn(x_train, y_train, params)
    y_prediction = model.predict(x_today)
    if params.binary_classification:
        y_prediction[y_prediction >= 0.5] = 2
        y_prediction[y_prediction < 0.5] = 0
        prediction = y_prediction[0][0]
    else:
        prediction = np.array([np.argmax(pred, axis=None, out=None) for pred in y_prediction])[0]
    if prediction == 0:
        prediction_str = 'sell'
    elif prediction == 2:
        prediction_str = 'buy'
    else:
        prediction_str = 'hold'
    return prediction, prediction_str


if __name__ == '__main__':
    reimport = True
    imp = Importer()
    imp.import_all(SYMBOLS, reimport)
    imp.import_all_technical_indicators(SYMBOLS)
    imp.process_data(reimport)
    if reimport:
        imp.export_to_csv_files('./../target/data')
    discrete_bench_params = NnBenchmarkParams(False)
    discrete_bench_params.walk_forward_testing = False
    discrete_bench_params.test_size = 0
    binary_bench_params = NnBenchmarkParams(True)
    binary_bench_params.walk_forward_testing = False
    binary_bench_params.test_size = 0

    df_list, sym_list = csv_importer.import_data_from_files(SYMBOLS, CSV_FILES_PATH)
    results_df = pd.DataFrame()
    last_day_idx_str = 'undefined'
    for symbol_it in range(0, len(SYMBOLS)):
        df = df_list[symbol_it]
        sym = sym_list[symbol_it]

        binary_bench_params.curr_sym = sym
        discrete_bench_params.curr_sym = sym
        last_day_idx = df.index.values[len(df) - 1]
        last_day_idx_str = pd.to_datetime(str(last_day_idx)).strftime('%Y-%m-%d')
        close_value = df.iloc[len(df) - 1][stock_constants.ADJUSTED_CLOSE_COL]
        train_df = df[(df.index < last_day_idx)]
        test_processed_df = benchmark_data_preprocessing.manage_and_drop_helper_df_columns(df.copy(),
                                                                                           discrete_bench_params.difference_non_stationary,
                                                                                           False)
        test_df = test_processed_df[(test_processed_df.index >= last_day_idx)]
        if len(test_df) == 0 or test_df.isnull().any(axis=1).iloc[0]:
            print("{0} CRITICAL ERR FOUND NULLS IN ROW".format(sym))
            continue

        binary_prediction, binary_prediction_str = teach_and_predict(binary_bench_params, train_df, test_df)
        discrete_prediction, discrete_prediction_str = teach_and_predict(discrete_bench_params, train_df, test_df)

        print(
            "{0} today close value is {1} prediction from {2} is binary ({3} == {4}) and discrete ({5} == {6}}})".format(
                sym, close_value,
                last_day_idx, binary_prediction, binary_prediction_str,
                discrete_prediction,
                discrete_prediction_str))
        result_dict = {'symbol': sym, 'date': last_day_idx, 'binary prediction value': binary_prediction,
                       'discrete prediction value': discrete_prediction,
                       'close price': close_value}
        results_df = results_df.append(result_dict, ignore_index=True)
    if results_df is not None and len(results_df) > 0:
        results_df.to_csv('{0}/predictions-{1}.csv'.format('./..', last_day_idx_str), index=False)
    print('Finished predicting')
