import math

import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.linear_model import LinearRegression

import alpha
import model_runner as model_runner
from plot_constants import *

api = alpha.AlphaVantage()
df = api.daily_adjusted('GOOGL')

df[alpha.HL_PCT_CHANGE_COL] = (df[alpha.HIGH_COL] - df[alpha.LOW_COL]) / df[alpha.LOW_COL] * 100.0
df[alpha.DAILY_PCT_CHANGE_COL] = ((df[alpha.CLOSE_COL] - df[alpha.OPEN_COL]) / df[alpha.CLOSE_COL]) * 100.0
df = df[[alpha.ADJUSTED_CLOSE_COL, alpha.VOLUME_COL, alpha.HL_PCT_CHANGE_COL, alpha.DAILY_PCT_CHANGE_COL]]
one_percent = int(math.ceil(0.01 * len(df)))
predictor = LinearRegression(n_jobs=-1)
df = model_runner.predict_exact(predictor, df, 1)

print(df.tail())

style.use('ggplot')
# df[alpha.ADJUSTED_CLOSE_COL].plot(label=CLOSE_PRICE_LABEL)
# df[alpha.FORECAST_COL].plot(label=FORECAST_LABEL)
# plt.legend(loc=4)
# plt.xlabel(DATE_LABEL)
# plt.ylabel(CLOSE_PRICE_USD_LABEL)
# plt.savefig('./../target/plot.png')
# plt.show()

#plt.close()
#

df[alpha.DAILY_PCT_CHANGE_COL].plot()
plt.legend(loc=4)
plt.xlabel(DATE_LABEL)
plt.ylabel(PRICE_CHANGE_PCT_LABEL)
plt.show()