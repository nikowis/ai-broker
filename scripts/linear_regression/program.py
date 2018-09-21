import math

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import alpha as stck
import model_runner as model_runner
from plot_constants import *

alpha = stck.AlphaVantage()
df = alpha.daily_adjusted(TICKER)


df = df[[alpha.OPEN_COL, alpha.HIGH_COL, alpha.LOW_COL, alpha.ADJUSTED_CLOSE_COL, alpha.VOLUME_COL]]

df = df[[alpha.ADJUSTED_CLOSE_COL]]
one_percent = int(math.ceil(0.01 * len(df)))
predictor = LinearRegression(n_jobs=-1)
df = model_runner.predict_exact(predictor, df, 1)

df[alpha.ADJUSTED_CLOSE_COL].plot(label=CLOSE_PRICE_LABEL)
df['Forecast'].plot(label=FORECAST_LABEL)
plt.legend(loc=4)
plt.xlabel(DATE_LABEL)
plt.ylabel(CLOSE_PRICE_USD_LABEL)
plt.title(TICKER)
plt.savefig('./../../target/linear_regression_1_day_full.png')

plt.close()

df = df[-30:]

df[alpha.ADJUSTED_CLOSE_COL].plot(label=CLOSE_PRICE_LABEL)
df['Forecast'].plot(label=FORECAST_LABEL)
plt.legend(loc=4)
plt.xlabel(DATE_LABEL)
plt.ylabel(CLOSE_PRICE_USD_LABEL)
plt.title(TICKER)
plt.savefig('./../../target/linear_regression_1_day_last_30.png')