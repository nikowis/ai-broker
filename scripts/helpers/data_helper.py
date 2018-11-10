import numpy as np
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler

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
        X_train = np.expand_dims(X_train, axis=1)
        X_test = np.expand_dims(X_test, axis=1)
        std_scale = StandardScaler().fit(X_train)
        X_train = std_scale.transform(X_train)
        X_test = std_scale.transform(X_test)
    return X_train, X_test, y_train, y_test


def prepare_label_extract_data(df, forecast_days):
    """
    Function stplitting dateframe data for machine learning.

    :param df: dateframe data
    :param forecast_days: how many days to forecast out
    :return:
        (df, X, y, X_lately) # tuple of (modified df, learning data, labels, data without labels)
    """

    X = np.array(df[const.ADJUSTED_CLOSE_COL])
    X_lately = X[-forecast_days:]
    X = X[:-forecast_days]
    df = df[:-forecast_days]
    df_removed = df.dropna()
    y = np.array(df_removed[const.LABEL_DISCRETE_COL])
    return df, X, y, X_lately
