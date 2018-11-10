import numpy as np
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler

import db.stock_constants as const


def train_test_split(X, y, scale=False):
    """
    Function for train test split.

    :param X: data
    :param y: labels
    :param scale: scale the data according to minmax scaler using train data only
    :return:
        (X_train, X_test, y_train, y_test) # Tuple
    """
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
    if scale:
        expanded_y_train = np.expand_dims(y_train, axis=1)
        expanded_y_test = np.expand_dims(y_test, axis=1)
        scaler = MinMaxScaler()
        train_data = np.append(X_train, expanded_y_train, axis=1)
        test_data = np.append(X_test, expanded_y_test, axis=1)
        scaler.fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)

        X_size = X_train.shape[1]

        X_train = train_data[:, :X_size]
        y_train = train_data[:, X_size]
        X_test = test_data[:, :X_size]
        y_test = test_data[:, X_size]
    return X_train, X_test, y_train, y_test


def prepare_label_extract_data(df, forecast_days):
    """
    Function stplitting dateframe data for machine learning.

    :param df: Dateframe data to modify.
    :param forecast_days: how many days to forecast out
    :return:
        (df, X, y, X_lately) # Tuple of (modified dateframe with label column, learning data, labels, data without labels)
    """
    df = df.copy()

    X = np.array(df[const.ADJUSTED_CLOSE_COL])

    X_lately = X[-forecast_days:]
    X = X[:-forecast_days]
    df = df[:-forecast_days]
    df_removed = df.dropna()
    y = np.array(df_removed[const.LABEL_DISCRETE_COL])
    return df, X, y, X_lately