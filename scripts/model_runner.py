import numpy as np
from sklearn import preprocessing, model_selection, svm
import datetime
import matplotlib.pyplot as plt
from matplotlib import style


def predict(forecast_col, predictor, dateframe, forecast_days, plot=False):
    df = dateframe.copy()
    df.fillna(value=-99999, inplace=True)
    df['label'] = df[forecast_col].shift(-forecast_days)

    X = np.array(df.drop(['label'], 1))
    # X = preprocessing.scale(X)
    X_lately = X[-forecast_days:]
    X = X[:-forecast_days]

    df.dropna(inplace=True)

    y = np.array(df['label'])

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
    predictor.fit(X_train, y_train)
    confidence = predictor.score(X_test, y_test)

    forecast_set = predictor.predict(X_lately)

    print('Confidence {}'.format(confidence))

    style.use('ggplot')
    df['Forecast'] = np.nan

    last_date = df.iloc[-1].name
    last_unix = last_date.timestamp()
    one_day_seconds = 86400
    next_unix = last_unix + one_day_seconds

    for i in forecast_set:
        next_date = datetime.datetime.utcfromtimestamp(next_unix)
        next_unix += one_day_seconds
        df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

    if plot:
        df[forecast_col].plot()
        df['Forecast'].plot()
        plt.legend(loc=4)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.show()
    return df
