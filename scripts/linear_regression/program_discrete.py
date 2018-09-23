import os

import matplotlib.pyplot as plt
from matplotlib import style
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

predictor = LinearRegression(n_jobs=-1)
df = model_runner.predict_rise_fall_stay(predictor, df, 1)

style.use('ggplot')
fig, ax = plt.subplots()

df.dropna(inplace=True)
df[alpha.LABEL_DISCRETE_COL].plot(kind='hist', xticks=[-1, 0, 1], label=RATE_CHANGE_LABEL)
df[alpha.FORECAST_DISCRETE_COL].plot(kind='hist', xticks=[-1, 0, 1], label=RATE_CHANGE_FORECAST_LABEL)
plt.legend(loc=2)
plt.xticks([-1, 0, 1], [FALL_LABEL, IDLE_LABEL, RISE_LABEL])
plt.xlabel(VALUE_CHANGE_LABEL)
plt.ylabel(FORECAST_COUNT_LABEL)
plt.title(TICKER)
plt.savefig('%s/l_r_discrete_score.eps' % base_path, format='eps', dpi=1000)
plt.savefig('%s/l_r_discrete_score.png' % base_path)
plt.show()
