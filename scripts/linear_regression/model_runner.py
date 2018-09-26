import datetime
import math

import numpy as np
from sklearn import model_selection

import alpha

FALL_VALUE = -1
IDLE_VALUE = 0
RISE_VALUE = 1


def predict_exact(predictor, dateframe, forecast_days):
    df = dateframe.copy()
    df.dropna(inplace=True)
    df.fillna(value=-99999, inplace=True)
    df[alpha.LABEL_COL] = df[alpha.ADJUSTED_CLOSE_COL].shift(-forecast_days)

    X = np.array(df.drop([alpha.LABEL_COL], 1))
    # X = preprocessing.scale(X)
    X_lately = X[-forecast_days:]
    X = X[:-forecast_days]

    df_removed = df.dropna()

    y = np.array(df_removed[alpha.LABEL_COL])

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
    predictor.fit(X_train, y_train)
    confidence = predictor.score(X_test, y_test)

    forecast_set = predictor.predict(X_lately)

    print('Confidence exact {}'.format(confidence))

    df[alpha.FORECAST_FOR_TODAY_COL] = np.nan

    last_date = df.iloc[-1].name
    last_unix = last_date.timestamp()
    one_day_seconds = 86400
    next_unix = last_unix + one_day_seconds

    for i in forecast_set:
        next_date = datetime.datetime.utcfromtimestamp(next_unix)
        next_unix += one_day_seconds
        df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

    full_predictions = predictor.predict(X)
    df[alpha.FORECAST_FOR_TODAY_COL][forecast_days:-forecast_days] = full_predictions

    return df


def predict_rise_fall_stay(predictor, dateframe, forecast_days, stay_tresh=0.4):
    df = dateframe.copy()
    df.dropna(inplace=True)
    df.fillna(value=-99999, inplace=True)

    df[alpha.LABEL_COL] = df[alpha.ADJUSTED_CLOSE_COL].shift(-forecast_days)

    df[alpha.DAILY_PCT_CHANGE_COL] = (df[alpha.LABEL_COL] - df[alpha.ADJUSTED_CLOSE_COL]) / df[
        alpha.ADJUSTED_CLOSE_COL] * 100.0

    df[alpha.LABEL_DISCRETE_COL] = df[alpha.DAILY_PCT_CHANGE_COL].apply(
        lambda pct: FALL_VALUE if pct < -stay_tresh else RISE_VALUE if pct > stay_tresh else IDLE_VALUE)

    X = np.array(df.drop([alpha.LABEL_COL, alpha.LABEL_DISCRETE_COL, alpha.DAILY_PCT_CHANGE_COL], 1))
    X_lately = X[-forecast_days:]
    X = X[:-forecast_days]

    df_removed = df.dropna()

    y = np.array(df_removed[alpha.LABEL_COL])

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
    predictor.fit(X_train, y_train)
    confidence = predictor.score(X_test, y_test)

    forecast_set = predictor.predict(X_lately)

    df[alpha.FORECAST_FOR_TODAY_COL] = np.nan

    last_date = df.iloc[-1].name
    last_unix = last_date.timestamp()
    one_day_seconds = 86400
    next_unix = last_unix + one_day_seconds

    for i in forecast_set:
        next_date = datetime.datetime.utcfromtimestamp(next_unix)
        next_unix += one_day_seconds
        df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

    full_predictions = predictor.predict(X)
    df[alpha.FORECAST_FOR_TODAY_COL][forecast_days:-forecast_days] = full_predictions

    df[alpha.FORECAST_FUTURE_COL] = df[alpha.FORECAST_FOR_TODAY_COL].shift(-forecast_days)
    df[alpha.FORECAST_PCT_CHANGE_COL] = (df[alpha.FORECAST_FUTURE_COL] - df[alpha.ADJUSTED_CLOSE_COL]) / df[
        alpha.ADJUSTED_CLOSE_COL] * 100.0
    df[alpha.FORECAST_DISCRETE_COL] = df[alpha.FORECAST_PCT_CHANGE_COL].apply(
        lambda pct: float('nan') if math.isnan(
            pct) else FALL_VALUE if pct < -stay_tresh else RISE_VALUE if pct > stay_tresh else IDLE_VALUE)

    confidence = 0
    for index, row in df.iterrows():
        if row[alpha.FORECAST_DISCRETE_COL] == row[alpha.LABEL_DISCRETE_COL]:
            confidence += 1
    confidence = confidence / len(df)
    print('Confidence discrete {}'.format(confidence))
    return df
