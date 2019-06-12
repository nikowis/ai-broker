import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style

from data_import import db_access
import stock_constants as const

MIN_DATE = '1900-01-01'
MAX_DATE = '2020-10-29'
SELECTED_SYM = 'GOOGL'
IMG_PATH = const.TARGET_DIR + '/data_analyzis/'
if not os.path.exists(IMG_PATH):
    os.makedirs(IMG_PATH)


def plot_columns(df):
    style.use('ggplot')

    line_plot_column(df, const.ADJUSTED_CLOSE_COL, 'GOOGL', 'Cena zamknięcia (USD)', 'Data')
    df[const.ADJUSTED_CLOSE_COL + ' stationary'] = df[const.ADJUSTED_CLOSE_COL].diff().fillna(0)
    line_plot_column(df, const.ADJUSTED_CLOSE_COL + ' stationary', SELECTED_SYM, 'Cena zamknięcia (USD)', 'Data')
    line_plot_column(df, const.VOLUME_COL, SELECTED_SYM, 'Liczba akcji w obrocie', 'Data')
    line_plot_column(df, const.HL_PCT_CHANGE_COL, SELECTED_SYM, 'Stosunek high/low (%)', 'Data')
    line_plot_column(df, const.SMA_5_COL, SELECTED_SYM, 'SMA-5', 'Data')
    line_plot_column(df, const.SMA_DIFF_COL, SELECTED_SYM, 'SMA-DIFF', 'Data')
    line_plot_column(df, const.TR_COL, SELECTED_SYM, 'TR', 'Data')
    line_plot_column(df, const.MACD_COL, SELECTED_SYM, 'MACD', 'Data')
    line_plot_column(df, const.MACD_SIGNAL_COL, SELECTED_SYM, 'MACD Signal', 'Data')

    line_plot_column(df, const.ROC_5_COL, SELECTED_SYM, 'ROC-5', 'Data')
    line_plot_column(df, const.ROC_DIFF_COL, SELECTED_SYM, 'ROC-DIFF', 'Data')

    line_plot_column(df, const.MOM_5_COL, SELECTED_SYM, 'MOM-5', 'Data')
    line_plot_column(df, const.MOM_DIFF_COL, SELECTED_SYM, 'MOM-DIFF', 'Data')

    line_plot_column(df, const.WILLR_5_COL, SELECTED_SYM, 'WILLR-5', 'Data')
    line_plot_column(df, const.WILLR_DIFF_COL, SELECTED_SYM, 'WILLR-DIFF', 'Data')

    line_plot_column(df, const.RSI_5_COL, SELECTED_SYM, 'RSI-5', 'Data')
    line_plot_column(df, const.RSI_DIFF_COL, SELECTED_SYM, 'RSI-DIFF', 'Data')

    line_plot_column(df, const.ADX_5_COL, SELECTED_SYM, 'ADX-5', 'Data')
    line_plot_column(df, const.ADX_DIFF_COL, SELECTED_SYM, 'ADX-DIFF', 'Data')

    line_plot_column(df, const.CCI_5_COL, SELECTED_SYM, 'CCI-5', 'Data')
    line_plot_column(df, const.CCI_DIFF_COL, SELECTED_SYM, 'CCI-DIFF', 'Data')

    line_plot_column(df, const.AD_COL, SELECTED_SYM, 'AD', 'Data')
    line_plot_column(df, const.STOCH_K_COL, SELECTED_SYM, 'STOCH %K', 'Data')
    line_plot_column(df, const.STOCH_D_COL, SELECTED_SYM, 'STOCH %D', 'Data')
    line_plot_column(df, const.STOCH_K_DIFF_COL, SELECTED_SYM, 'STOCH %K DIFF', 'Data')
    line_plot_column(df, const.STOCH_D_DIFF_COL, SELECTED_SYM, 'STOCH %D DIFF', 'Data')
    line_plot_column(df, const.DISPARITY_5_COL, SELECTED_SYM, 'DISPARITY 5', 'Data')
    line_plot_column(df, const.BBANDS_10_DIFF_COL, SELECTED_SYM, 'BBANDS 10 DIFF', 'Data')
    line_plot_column(df, const.PRICE_BBANDS_UP_10_COL, SELECTED_SYM, 'PRICE TO BBADS UP 10', 'Data')
    line_plot_column(df, const.PRICE_BBANDS_LOW_10_COL, SELECTED_SYM, 'PRICE TO BBADS LOW 10', 'Data')

    hist_plot_column(df, const.LABEL_BINARY_COL, SELECTED_SYM, 'Liczba przypadków', 'Trend', [-1, 1],
                     ['Maleje', 'Rośnie'])
    hist_plot_column(df, const.LABEL_DISCRETE_COL, SELECTED_SYM, 'Liczba przypadków', 'Trend', [-1, 0, 1],
                     ['Maleje', 'Utrzymuje się', 'Rośnie'])


def line_plot_column(df, colname, title, ylabel, xlabel):
    df[colname].plot(kind='line')
    describe_plot_and_save(colname, title, ylabel, xlabel)


def hist_plot_column(df, colname, title, ylabel, xlabel, xticks, tick_labels):
    df[colname].plot(kind='hist', xticks=xticks)
    plt.xticks(xticks, tick_labels)
    describe_plot_and_save(colname, title, ylabel, xlabel)


def describe_plot_and_save(colname, title, ylabel, xlabel):
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig('{}/{}.png'.format(IMG_PATH, colname))
    plt.savefig('{}/{}.pdf'.format(IMG_PATH, colname), format='pdf', dpi=1000)
    plt.close()


def describe_df():
    feature_names = list(df.columns.values)
    describe = df.describe(percentiles=[.01, 0.05, .5, .95, .99]).T
    describe = describe.drop(columns=['count'])
    describe.insert(loc=0, column='feature', value=feature_names)
    pd.options.display.float_format = '{:5,.2f}'.format
    print(describe.to_latex(index=False, longtable=True, ))


if __name__ == '__main__':
    db_conn = db_access.create_db_connection(remote=False)
    df_list, sym_list = db_access.find_by_tickers_to_dateframe_parse_to_df_list(db_conn, [SELECTED_SYM],
                                                                                min_date=MIN_DATE, max_date=MAX_DATE)

    df = df_list[0]
    describe_df()
    # plot_columns(df)
