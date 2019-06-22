import numpy as np
from keras.utils import to_categorical
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder

import csv_importer
import stock_constants as const
from benchmark_params import BenchmarkParams

CORRELATED_COLS = [const.APO_10_COL, const.APO_DIFF_COL, const.MOM_5_COL, const.MOM_10_COL, const.MOM_DIFF_COL,
                   const.ROC_5_COL,
                   const.ROC_10_COL, const.RSI_10_COL]

HELPER_COLS = [const.LABEL_COL, const.LABEL_BINARY_COL, const.LABEL_DISCRETE_COL, const.DAILY_PCT_CHANGE_COL,
               const.DIVIDENT_AMOUNT_COL, const.SPLIT_COEFFICIENT_COL, const.CLOSE_COL, const.BBANDS_10_RLB_COL,
               const.BBANDS_10_RMB_COL, const.BBANDS_10_RUB_COL, const.BBANDS_20_RLB_COL, const.BBANDS_20_RMB_COL,
               const.BBANDS_20_RUB_COL, const.MACD_HIST_COL]

SELECTED_SYM = 'GOOGL'


def preprocess(df, benchmark_params: BenchmarkParams):
    df_copy = df.copy()
    if benchmark_params.difference_non_stationary:
        df_copy[const.ADJUSTED_CLOSE_COL] = df_copy[const.ADJUSTED_CLOSE_COL].diff()
        df_copy[const.OPEN_COL] = df_copy[const.OPEN_COL].diff()
        df_copy[const.CLOSE_COL] = df_copy[const.CLOSE_COL].diff()
        df_copy[const.HIGH_COL] = df_copy[const.HIGH_COL].diff()
        df_copy[const.LOW_COL] = df_copy[const.LOW_COL].diff()
        df_copy[const.SMA_5_COL] = df_copy[const.SMA_5_COL].diff()
        df_copy[const.SMA_10_COL] = df_copy[const.SMA_10_COL].diff()
        df_copy[const.SMA_20_COL] = df_copy[const.SMA_20_COL].diff()

    df_copy.dropna(inplace=True)
    df_without_helper_cols = df_copy.drop(HELPER_COLS, axis=1)

    df_without_corelated_features = df_without_helper_cols.drop(CORRELATED_COLS, axis=1)

    if benchmark_params.binary_classification:
        y = np.array(df_copy[const.LABEL_BINARY_COL])
    else:
        y = np.array(df_copy[const.LABEL_DISCRETE_COL])
    x = np.array(df_without_corelated_features)

    if benchmark_params.binary_classification:
        encoder = LabelEncoder()
        encoded_y = encoder.fit_transform(y)
    else:
        encoded_y = to_categorical(y)

    if benchmark_params.walk_forward_testing:
        x_trains_list = []
        y_trains_list = []
        x_tests_list = []
        y_tests_list = []
        test_size = benchmark_params.test_size
        test_window_size = benchmark_params.walk_forward_test_window_size
        train_window_size = benchmark_params.max_train_window_size

        full_windows_count = int((test_size * len(x) / test_window_size))
        for i in range(0, full_windows_count + 1):
            train_end_idx = int((1 - test_size) * len(x) + i * test_window_size)
            if train_window_size is None:
                train_start_idx = 0
            else:
                train_start_idx = int(max(0, train_end_idx - train_window_size))
            test_start_idx = int(train_end_idx + 1)
            test_end_idx = int(train_end_idx + test_window_size)

            x_train = x[train_start_idx:train_end_idx + 1]
            y_train = encoded_y[train_start_idx:train_end_idx + 1]
            x_test = x[test_start_idx:test_end_idx + 1]
            y_test = encoded_y[test_start_idx:test_end_idx + 1]

            x_train, x_test = standardize_and_pca(benchmark_params, x_train, x_test)

            x_trains_list.append(x_train)
            y_trains_list.append(y_train)
            x_tests_list.append(x_test)
            y_tests_list.append(y_test)
        x_train = x_trains_list
        y_train = y_trains_list
        x_test = x_tests_list
        y_test = y_tests_list
    else:
        x_train, x_test, y_train, y_test = model_selection.train_test_split(x, encoded_y,
                                                                            test_size=benchmark_params.test_size,
                                                                            shuffle=False)
        if benchmark_params.max_train_window_size is not None and benchmark_params.max_train_window_size < \
                x_train.shape[0]:
            row_count = x_train.shape[0]
            x_train = x_train[row_count - benchmark_params.max_train_window_size:, :]
            if benchmark_params.binary_classification:
                y_train = y_train[row_count - benchmark_params.max_train_window_size:]
            else:
                y_train = y_train[row_count - benchmark_params.max_train_window_size:, :]

        x_train, x_test = standardize_and_pca(benchmark_params, x_train, x_test)

    return x, y, x_train, x_test, y_train, y_test


def standardize_and_pca(preprocessing_params, x_train, x_test):
    if preprocessing_params.standardize:
        scale = StandardScaler().fit(x_train)
        x_train = scale.transform(x_train)
        x_test = scale.transform(x_test)
    if preprocessing_params.pca is not None and not preprocessing_params.walk_forward_testing:
        pca = PCA(preprocessing_params.pca).fit(x_train)
        x_train = pca.transform(x_train)
        x_test = pca.transform(x_test)
    return x_train, x_test


if __name__ == '__main__':
    df_list, _ = csv_importer.import_data_from_files([SELECTED_SYM])
    df = df_list[0]
