import random

import keras
import numpy as np
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler

import stock_constants as const


def extract_data(df, history_days=0, binary_classification=False):
    """
    Function stplitting dateframe data for machine learning.

    :param df: dateframe data
    :param history_days: how many days from history to put in each row
    :return:
        (df, x_standardized, x_train, x_test, y_train_one_hot, y_test_one_hot) # tuple of (modified df, standardized x,
        learning x, testing x, learning y (one hot), test y (one hot))
    """

    df, x = get_x_columns(df)
    df, x = calculate_history_columns(df, x, history_days)
    if binary_classification:
        y = np.array(df[const.LABEL_BINARY_COL])
    else:
        y = np.array(df[const.LABEL_DISCRETE_COL])

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, shuffle=False)

    if len(x_train.shape) == 1:
        x_train = np.expand_dims(x_train, axis=1)
        x_test = np.expand_dims(x_test, axis=1)
    std_scale = StandardScaler().fit(x_train)
    x_train = std_scale.transform(x_train)
    x_test = std_scale.transform(x_test)
    x_standardized = std_scale.transform(x)
    y_train_one_hot = keras.utils.to_categorical(y_train)
    y_test_one_hot = keras.utils.to_categorical(y_test)

    return df, x_standardized, x_train, x_test, y_train_one_hot, y_test_one_hot


def extract_data_from_list(df_list, history_days=0, test_size=0.2, binary_classification=False):
    data_count = len(df_list)
    test_count = int(data_count * test_size)
    test_indices = random.sample(range(1, data_count), test_count)
    test_dfs = [df_list[i] for i in test_indices]
    train_dfs = [i for j, i in enumerate(df_list) if j not in test_indices]

    total_x_train = None
    total_x_test = None
    total_y_train = None
    total_y_test = None

    for train_df in train_dfs:
        train_df, total_x_train, total_y_train = calculate_append_x_y(history_days, total_x_train,
                                                                      total_y_train,
                                                                      train_df, binary_classification)
    for test_df in test_dfs:
        test_df, total_x_test, total_y_test = calculate_append_x_y(history_days, total_x_test,
                                                                   total_y_test,
                                                                   test_df, binary_classification)

    y_train_one_hot = keras.utils.to_categorical(total_y_train)
    y_test_one_hot = keras.utils.to_categorical(total_y_test)
    return total_x_train, total_x_test, y_train_one_hot, y_test_one_hot


def calculate_history_columns(df, x, history_days):
    if history_days > 0:
        input_rows = x.shape[0]
        input_columns = x.shape[1]
        extended_x = np.zeros((input_rows, input_columns + history_days * input_columns))
        extended_x[:, :] = np.nan
        extended_x[:, 0:input_columns] = x
        for i in range(1, history_days + 1):
            extended_x[i:, i * input_columns:(i + 1) * input_columns] = x[:input_rows - i, :]
        nan_row_count = np.count_nonzero(np.isnan(extended_x).any(axis=1))
        x = extended_x[~np.isnan(extended_x).any(axis=1)]
        df = df[nan_row_count:]
    x = x[:-const.FORECAST_DAYS]
    df = df[:-const.FORECAST_DAYS]
    return df, x


def calculate_append_x_y(history_days, total_x, total_y, df, binary_classification):
    df, x = get_x_columns(df)
    df, x = calculate_history_columns(df, x, history_days)
    if binary_classification:
        y = np.array(df[const.LABEL_BINARY_COL])
    else:
        y = np.array(df[const.LABEL_DISCRETE_COL])
    std_scale = StandardScaler().fit(x)
    x = std_scale.transform(x)

    # first iteration
    if total_x is None:
        total_x = x
        total_y = y
    else:
        # append to existing numpy arrays
        total_x_rows = total_x.shape[0]
        total_x_cols = total_x.shape[1]
        total_y_rows = total_y.shape[0]
        extended_total_x_train = np.zeros((total_x_rows + x.shape[0], total_x_cols))
        extended_total_y_train = np.zeros((total_y_rows + y.shape[0]))
        extended_total_x_train[0:total_x_rows, :] = total_x
        extended_total_x_train[total_x_rows:, :] = x
        extended_total_y_train[0:total_y_rows] = total_y
        extended_total_y_train[total_y_rows:] = y
        total_x = extended_total_x_train
        total_y = extended_total_y_train
    return df, total_x, total_y


def get_x_columns(df):
    df[const.ADJUSTED_CLOSE_COL] = df[const.ADJUSTED_CLOSE_COL].diff().fillna(0)
    df[const.OPEN_COL] = df[const.OPEN_COL].diff().fillna(0)
    df[const.HIGH_COL] = df[const.HIGH_COL].diff().fillna(0)
    df[const.LOW_COL] = df[const.LOW_COL].diff().fillna(0)
    df[const.VOLUME_COL] = df[const.VOLUME_COL].diff().fillna(0)
    df[const.SMA_10_COL] = df[const.SMA_10_COL].diff().fillna(0)
    df[const.SMA_20_COL] = df[const.SMA_20_COL].diff().fillna(0)
    df[const.EMA_10_COL] = df[const.EMA_10_COL].diff().fillna(0)
    df[const.EMA_20_COL] = df[const.EMA_20_COL].diff().fillna(0)

    # x = np.array(df[[const.ADJUSTED_CLOSE_COL, const.VOLUME_COL, const.OPEN_COL, const.HL_PCT_CHANGE_COL]])
    x = np.array(df[[const.ADJUSTED_CLOSE_COL]])
    return df, x
