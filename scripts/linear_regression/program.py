import math
import os

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import alpha
import model_runner as model_runner
from plot_constants import *

TICKER = 'GOOGL'

api = alpha.AlphaVantage()
df = api.daily_adjusted(TICKER)

base_path = './../../target/linear_regression'

if not os.path.exists(base_path):
    os.makedirs(base_path)

df = df[[alpha.ADJUSTED_CLOSE_COL]]
one_percent = int(math.ceil(0.01 * len(df)))
predictor = LinearRegression(n_jobs=-1)
df = model_runner.predict_exact(predictor, df, 1)

df[alpha.ADJUSTED_CLOSE_COL].plot(label=CLOSE_PRICE_LABEL)
plt.legend(loc=4)
plt.xlabel(DATE_LABEL)
plt.ylabel(CLOSE_PRICE_USD_LABEL)
plt.title(TICKER)

plt.savefig('%s/l_r_stock_data.eps' % base_path, format='eps', dpi=1000)
plt.savefig('%s/l_r_stock_data.png' % base_path)

plt.close()

df[alpha.ADJUSTED_CLOSE_COL].plot(label=CLOSE_PRICE_LABEL)
df['Forecast'].plot(label=FORECAST_LABEL)
plt.legend(loc=4)
plt.xlabel(DATE_LABEL)
plt.ylabel(CLOSE_PRICE_USD_LABEL)
plt.title(TICKER)
plt.savefig('%s/l_r_1_day_full.eps' % base_path, format='eps', dpi=1000)
plt.savefig('%s/l_r_1_day_full.png' % base_path)
plt.close()

df = df[-30:]

df[alpha.ADJUSTED_CLOSE_COL].plot(label=CLOSE_PRICE_LABEL)
df['Forecast'].plot(label=FORECAST_LABEL)
plt.legend(loc=4)
plt.xlabel(DATE_LABEL)
plt.ylabel(CLOSE_PRICE_USD_LABEL)
plt.title(TICKER)
plt.savefig('%s/l_r_1_day_last_30.eps' % base_path, format='eps', dpi=1000)
plt.savefig('%s/l_r_1_day_last_30.png' % base_path)
