import math

import numpy as np

from helpers import alpha, data_helper

FALL_VALUE = -1
IDLE_VALUE = 0
RISE_VALUE = 1


def predict_exact(predictor, dateframe, forecast_days):
    df, X, y, X_lately = data_helper.prepare_label_extract_data(dateframe, forecast_days)
    X_train, X_test, y_train, y_test = data_helper.train_test_split(X, y)
    predictor.fit(X_train, y_train)
    confidence = predictor.score(X_test, y_test)
    df[alpha.FORECAST_FOR_TODAY_COL] = np.nan
    full_predictions = predictor.predict(X)
    df[alpha.FORECAST_FOR_TODAY_COL][forecast_days:] = full_predictions[:-forecast_days]

    print('Confidence exact {}'.format(confidence))

    return df


def predict_discrete(predictor, dateframe, forecast_days, stay_tresh=0.4):
    df = predict_exact(predictor, dateframe, forecast_days)
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
