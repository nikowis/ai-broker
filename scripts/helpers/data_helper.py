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
    :return:
        (df, X, y, X_lately) # tuple of (modified df, learning data, labels, data without labels)
    """
    df[const.HL_PCT_CHANGE_COL] = (df[const.HIGH_COL] - df[const.LOW_COL]) / df[const.HIGH_COL] * 100
    X = np.array(df[[const.VOLUME_COL, const.ADJUSTED_CLOSE_COL, const.HL_PCT_CHANGE_COL]])
    if history_days > 0:
        input_rows = X.shape[0]
        input_columns = X.shape[1]
        extended_X = np.zeros((input_rows, input_columns + history_days * input_columns))
        extended_X[:, :] = np.nan
        extended_X[:, 0:input_columns] = X
        for i in range(1, history_days + 1):
            extended_X[i:, i * input_columns:(i + 1) * input_columns] = X[:input_rows - i, :]
        nan_row_count = np.count_nonzero(np.isnan(extended_X).any(axis=1))
        X = extended_X[~np.isnan(extended_X).any(axis=1)]
        df = df[nan_row_count:]

    X_lately = X[-forecast_days:]
    X = X[:-forecast_days]
    df = df[:-forecast_days]
    y = np.array(df[const.LABEL_DISCRETE_COL])
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
    X_standarized, X_test, X_train = standardize(X, X_test, X_train)
    y_train_binary = keras.utils.to_categorical(y_train)
    y_test_binary = keras.utils.to_categorical(y_test)
    return df, X_standarized, X_train, X_test, y_train_binary, y_test_binary


def standardize(X, X_test, X_train):
    # standarize
    if len(X_train.shape) == 1:
        X_train = np.expand_dims(X_train, axis=1)
        X_test = np.expand_dims(X_test, axis=1)
    std_scale = StandardScaler().fit(X_train)
    X_train = std_scale.transform(X_train)
    X_test = std_scale.transform(X_test)
    X_standarized = std_scale.transform(X)
    return X_standarized, X_test, X_train
