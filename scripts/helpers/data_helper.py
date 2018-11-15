import keras
import numpy as np
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler

import db.stock_constants as const


def extract_data(df, forecast_days=1, history_days=0):
    """
    Function stplitting dateframe data for machine learning.

    :param df: dateframe data
    :param forecast_days: how many days to forecast out
    :param history_days: how many days from history to put in each row
    :return:
        (df, x_standardized, x_train, x_test, y_train_binary, y_test_binary) # tuple of (modified df, standardized x,
        learning x, testing x, learning y (one hot), test y (one hot))
    """
    df[const.HL_PCT_CHANGE_COL] = (df[const.HIGH_COL] - df[const.LOW_COL]) / df[const.HIGH_COL] * 100
    x = np.array(df[[const.VOLUME_COL, const.ADJUSTED_CLOSE_COL, const.HL_PCT_CHANGE_COL]])
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

    x_lately = x[-forecast_days:]
    x = x[:-forecast_days]
    df = df[:-forecast_days]
    y = np.array(df[const.LABEL_DISCRETE_COL])
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)
    x_standardized, x_test, x_train = standardize(x, x_test, x_train)
    y_train_binary = keras.utils.to_categorical(y_train)
    y_test_binary = keras.utils.to_categorical(y_test)
    return df, x_standardized, x_train, x_test, y_train_binary, y_test_binary


def standardize(x, x_test, x_train):
    # standardize
    if len(x_train.shape) == 1:
        x_train = np.expand_dims(x_train, axis=1)
        x_test = np.expand_dims(x_test, axis=1)
    std_scale = StandardScaler().fit(x_train)
    x_train = std_scale.transform(x_train)
    x_test = std_scale.transform(x_test)
    x_standardized = std_scale.transform(x)
    return x_standardized, x_test, x_train
