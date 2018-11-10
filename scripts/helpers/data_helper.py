import numpy as np

import db.stock_constants as const


def prepare_label_extract_data(df, forecast_days):
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
    return df, X, y, X_lately
