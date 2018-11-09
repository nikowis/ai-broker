import os

from sklearn.linear_model import LinearRegression
from matplotlib import style
from data_processing import alpha
from deprecated_code.linear_regression import lr_model_runner as model_runner
from helpers.plot_helper import *

TICKER = 'GOOGL'

api = alpha.AlphaVantage()
df = api.data(TICKER)

base_path = './../../target/linear_regression'

if not os.path.exists(base_path):
    os.makedirs(base_path)

df = df[[alpha.ADJUSTED_CLOSE_COL]]

predictor = LinearRegression(n_jobs=-1)
df = model_runner.predict_exact(predictor, df, 1)

df[alpha.DAILY_PCT_CHANGE_COL] = (df[alpha.LABEL_COL] - df[alpha.ADJUSTED_CLOSE_COL]) / df[
    alpha.ADJUSTED_CLOSE_COL] * 100.0

style.use('ggplot')

df[alpha.ADJUSTED_CLOSE_COL].plot(label=CLOSE_PRICE_LABEL)
legend_labels_save_files(TICKER, 'l_r_stock_data', base_path)

df[alpha.ADJUSTED_CLOSE_COL].plot(label=CLOSE_PRICE_LABEL)
df[alpha.FORECAST_FOR_TODAY_COL].plot(label=FORECAST_LABEL)
legend_labels_save_files(TICKER, 'l_r_1_day_full', base_path)

df_30 = df[-30:]

df_30[alpha.ADJUSTED_CLOSE_COL].plot(label=CLOSE_PRICE_LABEL)
df_30[alpha.FORECAST_FOR_TODAY_COL].plot(label=FORECAST_LABEL)
legend_labels_save_files(TICKER, 'l_r_1_day_last_30', base_path)

df_30 = df_30.copy()
df_30[alpha.DAILY_PCT_CHANGE_COL].plot()
legend_labels_save_files(TICKER, 'l_r_pct_change_last_30', base_path, ylabel=PRICE_CHANGE_PCT_LABEL, legend=-1)
