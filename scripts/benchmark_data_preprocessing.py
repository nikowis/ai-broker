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
    df_without_corelated_features = manage_and_drop_helper_df_columns(df_copy, benchmark_params.difference_non_stationary)
    df_without_corelated_features.dropna(inplace=True)
    df_copy.dropna(inplace=True)
    if benchmark_params.binary_classification:
        y = np.array(df_copy[const.LABEL_BINARY_COL])
    else:
        y = np.array(df_copy[const.LABEL_DISCRETE_COL])
    x = np.array(df_without_corelated_features)
    benchmark_params.feature_names = list(df_without_corelated_features.columns)

    if benchmark_params.binary_classification:
        encoder = LabelEncoder()
        encoded_y = encoder.fit_transform(y)
    else:
        if benchmark_params.one_hot_encode_labels:
            encoded_y = to_categorical(y)
        else:
            encoder = LabelEncoder()
            encoded_y = encoder.fit_transform(y)

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
            test_start_idx = int(train_end_idx)
            test_end_idx = int(train_end_idx + test_window_size)

            x_train = x[train_start_idx:train_end_idx]
            y_train = encoded_y[train_start_idx:train_end_idx]
            x_test = x[test_start_idx:test_end_idx + 1]
            y_test = encoded_y[test_start_idx:test_end_idx + 1]

            x_train, x_test, std_scaler, pca_transformer = standardize_and_pca(benchmark_params, x_train, x_test)

            x_trains_list.append(x_train)
            y_trains_list.append(y_train)
            x_tests_list.append(x_test)
            y_tests_list.append(y_test)
        x_train = x_trains_list
        y_train = y_trains_list
        x_test = x_tests_list
        y_test = y_tests_list
    else:
        if benchmark_params.test_size != 0:
            x_train, x_test, y_train, y_test = model_selection.train_test_split(x, encoded_y,
                                                                            test_size=benchmark_params.test_size,
                                                                            shuffle=False)
        else:
            x_train = x
            y_train = encoded_y
            x_test = None
            y_test = None
        if benchmark_params.max_train_window_size is not None and benchmark_params.max_train_window_size < \
                x_train.shape[0]:
            row_count = x_train.shape[0]
            x_train = x_train[row_count - benchmark_params.max_train_window_size:, :]
            if benchmark_params.binary_classification:
                y_train = y_train[row_count - benchmark_params.max_train_window_size:]
            else:
                y_train = y_train[row_count - benchmark_params.max_train_window_size:, :]

        x_train, x_test, std_scaler, pca_transformer = standardize_and_pca(benchmark_params, x_train, x_test)

    return x, y, x_train, x_test, y_train, y_test, std_scaler, pca_transformer


def manage_and_drop_helper_df_columns(df, difference_non_stationary=True):
    if difference_non_stationary:
        df[const.ADJUSTED_CLOSE_COL] = df[const.ADJUSTED_CLOSE_COL].diff()
        df[const.OPEN_COL] = df[const.OPEN_COL].diff()
        df[const.CLOSE_COL] = df[const.CLOSE_COL].diff()
        df[const.HIGH_COL] = df[const.HIGH_COL].diff()
        df[const.LOW_COL] = df[const.LOW_COL].diff()
        df[const.SMA_5_COL] = df[const.SMA_5_COL].diff()
        df[const.SMA_10_COL] = df[const.SMA_10_COL].diff()
        df[const.SMA_20_COL] = df[const.SMA_20_COL].diff()
    df_without_helper_cols = df.drop(HELPER_COLS, axis=1)
    df_without_corelated_features = df_without_helper_cols.drop(CORRELATED_COLS, axis=1)
    return df_without_corelated_features


def standardize_and_pca(preprocessing_params, x_train, x_test):
    std_scaler = None
    pca_transformer = None
    if preprocessing_params.standardize:
        std_scaler = StandardScaler().fit(x_train)
        x_train = std_scaler.transform(x_train)
        if x_test is not None:
            x_test = std_scaler.transform(x_test)
    if preprocessing_params.pca is not None:
        pca_transformer = PCA(preprocessing_params.pca).fit(x_train)
        x_train = pca_transformer.transform(x_train)
        if x_test is not None:
            x_test = pca_transformer.transform(x_test)
    return x_train, x_test, std_scaler, pca_transformer


if __name__ == '__main__':
    df_list, _ = csv_importer.import_data_from_files([SELECTED_SYM])
    df = df_list[0]
