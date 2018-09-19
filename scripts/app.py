from sklearn.linear_model import LinearRegression
import alpha as stck
import model_runner as model_runner
from sklearn import svm

import math

alpha = stck.AlphaVantage()
df = alpha.daily_adjusted('GOOGL')

PCT_CHANGE = 'PCT_change'
HL_PCT = 'HL_PCT'

#print(df.tail())

df = df[[alpha.OPEN_COL, alpha.HIGH_COL, alpha.LOW_COL, alpha.ADJUSTED_CLOSE_COL, alpha.VOLUME_COL]]

df[HL_PCT] = (df[alpha.HIGH_COL] - df[alpha.LOW_COL]) / df[alpha.LOW_COL] * 100.0
df[PCT_CHANGE] = (df[alpha.ADJUSTED_CLOSE_COL] - df[alpha.OPEN_COL]) / df[alpha.OPEN_COL] * 100.0
df = df[[alpha.ADJUSTED_CLOSE_COL, HL_PCT, PCT_CHANGE, alpha.VOLUME_COL]]
one_percent = int(math.ceil(0.01 * len(df)))
model_runner.predict(alpha.ADJUSTED_CLOSE_COL, LinearRegression(n_jobs=-1), df, one_percent, True)
#model_runner.predict(alpha.ADJUSTED_CLOSE_COL, svm.SVR(kernel='linear'), df, one_percent, True)
