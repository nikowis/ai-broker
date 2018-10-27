import os

from matplotlib import style
from sklearn.linear_model import LinearRegression

from helpers import alpha
from linear_regression import lr_model_runner as model_runner
from helpers.plot_helper import *

TICKER = 'GOOGL'

api = alpha.AlphaVantage()
df = api.data(TICKER)

base_path = './../../target/linear_regression'

if not os.path.exists(base_path):
    os.makedirs(base_path)

df = df[[alpha.ADJUSTED_CLOSE_COL]]

predictor = LinearRegression(n_jobs=-1)
df = model_runner.predict_discrete(predictor, df, 1)

style.use('ggplot')
fig, ax = plt.subplots()

df.dropna(inplace=True)
df[alpha.LABEL_DISCRETE_COL].plot(kind='hist', xticks=[-1, 0, 1], label=RATE_CHANGE_LABEL)
df[alpha.FORECAST_DISCRETE_COL].plot(kind='hist', xticks=[-1, 0, 1], label=RATE_CHANGE_FORECAST_LABEL)
plt.xticks([-1, 0, 1], [FALL_LABEL, IDLE_LABEL, RISE_LABEL])
legend_labels_save_files(TICKER, 'l_r_discrete_score', base_path, VALUE_CHANGE_LABEL, FORECAST_COUNT_LABEL, 2)

