import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
import keras
import db.stock_constants as const



def extract_data(df, forecast_days):
    """
    Function stplitting dateframe data for machine learning.

    :param df: dateframe data
    :param forecast_days: how many days to forecast out
    :return:
        (df, X, y, X_lately) # tuple of (modified df, learning data, labels, data without labels)
    """
    df[const.HL_PCT_CHANGE_COL] = (df[const.HIGH_COL] - df[const.LOW_COL]) / df[const.HIGH_COL] * 100
    X = np.array(df[[const.VOLUME_COL, const.ADJUSTED_CLOSE_COL, const.HL_PCT_CHANGE_COL]])
    X_lately = X[-forecast_days:]
    X = X[:-forecast_days]
    df = df[:-forecast_days]
    df_removed = df.dropna()
    y = np.array(df_removed[const.LABEL_DISCRETE_COL])
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
    X_standarized, X_test, X_train = standardize(X, X_test, X_train)
    y_train_binary = keras.utils.to_categorical(y_train)
    y_test_binary = keras.utils.to_categorical(y_test)
    return df, X_standarized,X_train, X_test, y_train_binary, y_test_binary



def standardize(X, X_test, X_train):
    #standarize
    if len(X_train.shape) == 1:
        X_train = np.expand_dims(X_train, axis=1)
        X_test = np.expand_dims(X_test, axis=1)
    std_scale = StandardScaler().fit(X_train)
    X_train = std_scale.transform(X_train)
    X_test = std_scale.transform(X_test)
    X_standarized = std_scale.transform(X)
    return X_standarized, X_test, X_train
