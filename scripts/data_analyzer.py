import matplotlib.pyplot as plt
from matplotlib import style
import os
import db_access
import stock_constants as const

MIN_DATE = '1900-01-01'
MAX_DATE = '2020-10-29'
SELECTED_SYM = 'GOOGL'
IMG_PATH = './../target/data_analyzis/'
if not os.path.exists(IMG_PATH):
    os.makedirs(IMG_PATH)

def main():
    db_conn = db_access.create_db_connection(remote=False)
    df_list, sym_list = db_access.find_by_tickers_to_dateframe_parse_to_df_list(db_conn, [SELECTED_SYM],
                                                                                min_date=MIN_DATE, max_date=MAX_DATE)

    df = df_list[0]
    style.use('ggplot')

    line_plot_column(df, const.ADJUSTED_CLOSE_COL, 'GOOGL', 'Cena zamknięcia (USD)', 'Data')
    df[const.ADJUSTED_CLOSE_COL+' stationary'] = df[const.ADJUSTED_CLOSE_COL].diff().fillna(0)
    line_plot_column(df, const.ADJUSTED_CLOSE_COL+' stationary', 'GOOGL', 'Cena zamknięcia (USD)', 'Data')

    hist_plot_column(df, const.LABEL_BINARY_COL, 'GOOGL', 'Liczba przypadków', 'Trend', [-1, 1], ['Maleje', 'Rośnie'])



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


if __name__ == '__main__':
    main()
