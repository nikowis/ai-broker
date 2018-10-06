import os

import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.linear_model import LinearRegression

import alpha
from linear_regression import model_runner as model_runner
from plot_constants import *



TICKER = 'GOOGL'

api = alpha.AlphaVantage()
df = api.data(TICKER)

base_path = './../../target/linear_regression'

if not os.path.exists(base_path):
    os.makedirs(base_path)

df = df[[alpha.ADJUSTED_CLOSE_COL]]

predictor = LinearRegression(n_jobs=-1)
df = model_runner.predict_exact(predictor, df, 1)

df[alpha.DAILY_PCT_CHANGE_COL] = (df[alpha.LABEL_COL] - df[alpha.ADJUSTED_CLOSE_COL]) / df[alpha.ADJUSTED_CLOSE_COL] * 100.0

style.use('ggplot')

df[alpha.ADJUSTED_CLOSE_COL].plot(label=CLOSE_PRICE_LABEL)
plt.legend(loc=4)
plt.xlabel(DATE_LABEL)
plt.ylabel(CLOSE_PRICE_USD_LABEL)
plt.title(TICKER)

plt.savefig('%s/l_r_stock_data.eps' % base_path, format='eps', dpi=1000)
plt.savefig('%s/l_r_stock_data.png' % base_path)
plt.show()

plt.close()

df[alpha.ADJUSTED_CLOSE_COL].plot(label=CLOSE_PRICE_LABEL)
df[alpha.FORECAST_FOR_TODAY_COL].plot(label=FORECAST_LABEL)
plt.legend(loc=4)
plt.xlabel(DATE_LABEL)
plt.ylabel(CLOSE_PRICE_USD_LABEL)
plt.title(TICKER)
plt.savefig('%s/l_r_1_day_full.eps' % base_path, format='eps', dpi=1000)
plt.savefig('%s/l_r_1_day_full.png' % base_path)
plt.show()
plt.close()

df_30 = df[-30:]

df_30[alpha.ADJUSTED_CLOSE_COL].plot(label=CLOSE_PRICE_LABEL)
df_30[alpha.FORECAST_FOR_TODAY_COL].plot(label=FORECAST_LABEL)
plt.legend(loc=4)
plt.xlabel(DATE_LABEL)
plt.ylabel(CLOSE_PRICE_USD_LABEL)
plt.title(TICKER)
plt.savefig('%s/l_r_1_day_last_30.eps' % base_path, format='eps', dpi=1000)
plt.savefig('%s/l_r_1_day_last_30.png' % base_path)
plt.show()
plt.close()

df_30 = df_30.copy()
df_30[alpha.DAILY_PCT_CHANGE_COL].plot(kind='line')
plt.xlabel(DATE_LABEL)
plt.ylabel(PRICE_CHANGE_PCT_LABEL)
plt.title(TICKER)
plt.savefig('%s/l_r_pct_change_last_30.eps' % base_path, format='eps', dpi=1000)
plt.savefig('%s/l_r_pct_change_last_30.png' % base_path)
plt.show()
plt.close()
