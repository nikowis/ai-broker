import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
import scripts.alpha as stck

alpha = stck.AlphaVantage()
df = alpha.daily_adjusted('GOOGL')

PCT_CHANGE = 'PCT_change'
HL_PCT = 'HL_PCT'

print(df.tail())

df = df[[alpha.OPEN_COL, alpha.HIGH_COL, alpha.LOW_COL, alpha.ADJUSTED_CLOSE_COL, alpha.VOLUME_COL]]

df[HL_PCT] = (df[alpha.HIGH_COL] - df[alpha.LOW_COL]) / df[alpha.LOW_COL] * 100.0
df[PCT_CHANGE] = (df[alpha.ADJUSTED_CLOSE_COL] - df[alpha.OPEN_COL]) / df[alpha.OPEN_COL] * 100.0
df = df[[alpha.ADJUSTED_CLOSE_COL, HL_PCT, PCT_CHANGE, alpha.VOLUME_COL]]

forecast_col = alpha.ADJUSTED_CLOSE_COL
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)

y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)

print(forecast_set, confidence, forecast_out)

style.use('ggplot')
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day_seconds = 86400
next_unix = last_unix + one_day_seconds

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day_seconds
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

df[alpha.ADJUSTED_CLOSE_COL].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
