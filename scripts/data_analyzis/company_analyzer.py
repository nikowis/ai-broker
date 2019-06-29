import os

import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.dates as mdates

import csv_importer
import stock_constants as const

TARGET_PATH = './../../target'
CSV_FILES_PATH = TARGET_PATH + '/data'
COMPANY_FILES_PATH = TARGET_PATH + '/company_info'

FORECAST_COUNT_LABEL = 'Liczba prognoz'
VALUE_CHANGE_LABEL = 'Zmiana wartości'
HISTOGRAM_TITLE = 'Histogram predykcji dla zbioru testowego'

RATE_CHANGE_LABEL = 'Zmiana kursu'
RISE_LABEL = 'wzrost'
IDLE_LABEL = 'utrzymanie'
FALL_LABEL = 'spadek'

if not os.path.exists(COMPANY_FILES_PATH):
    os.makedirs(COMPANY_FILES_PATH)


def plot_company_summary(df, symbol):
    plt.figure(figsize=(12, 12))
    style.use('ggplot')
    plt.suptitle(symbol)

    plt.subplot(2, 2, 1)
    df[const.ADJUSTED_CLOSE_COL].plot(kind='line')
    plt.title('Cena zamknięcia')
    plt.ylabel('cena')
    plt.xlabel('data')

    plt.subplot(2, 2, 2)
    df[const.DAILY_PCT_CHANGE_COL].plot(kind='line')
    plt.title('Zmiana % względem dnia następnego')
    plt.ylabel('%')
    plt.xlabel('data')

    plt.subplot(2, 2, 3)
    df[const.HL_PCT_CHANGE_COL] = (df[const.HIGH_COL] - df[const.LOW_COL]) / df[
        const.HIGH_COL] * 100
    df[const.HL_PCT_CHANGE_COL].plot(kind='line')
    plt.title('Procentowa zmiana H/L')
    plt.ylabel('%')
    plt.xlabel('data')

    plt.subplot(2, 2, 4)
    df[const.LABEL_DISCRETE_COL].plot(kind='hist', xticks=[0, 1, 2], label=RATE_CHANGE_LABEL)
    plt.xticks([0, 1, 2], [FALL_LABEL, IDLE_LABEL, RISE_LABEL])
    plt.xlabel(VALUE_CHANGE_LABEL)
    plt.ylabel(FORECAST_COUNT_LABEL)
    plt.title(HISTOGRAM_TITLE)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig('{}/{}.png'.format(COMPANY_FILES_PATH, symbol))

    # plt.show()
    plt.close()

def compare_adj_one_plot():
    symbols = ['GOOGL', 'AMZN']
    df_list, _ = csv_importer.import_data_from_files(symbols, CSV_FILES_PATH)

    balanced_syms = []

    for i in range(0, len(symbols)):
        sym = symbols[i]
        df = df_list[i]
        MIN_DATE = '1990-01-01'
        MAX_DATE = '2020-10-29'
        df = df[(df.index > MIN_DATE)]
        df = df[(df.index < MAX_DATE)]
        style.use('ggplot')
        df[const.ADJUSTED_CLOSE_COL].plot(kind='line', x_compat=True, label='Cena zamknięcia ' + sym)
    # plt.title(sym)
    plt.ylabel('Cena zamknięcia (USD)')
    plt.xlabel('Data')
    plt.legend()
    # plt.show()
    plt.savefig('{}/{}.pdf'.format(COMPANY_FILES_PATH, 'close_comparison'),
                format='pdf', dpi=1000)
    plt.savefig('{}/{}.png'.format(COMPANY_FILES_PATH, 'close_comparison'))
    plt.close()

def plot_close_line_and_label_histograms():
    symbols = ['GOOGL', 'INTC', 'FB']
    df_list, _ = csv_importer.import_data_from_files(symbols, CSV_FILES_PATH)

    balanced_syms = []

    for i in range(0, len(symbols)):
        sym = symbols[i]
        df = df_list[i]
        MIN_DATE = '2010-01-01'
        MAX_DATE = '2020-10-29'
        df = df[(df.index > MIN_DATE)]
        df = df[(df.index < MAX_DATE)]

        # bincount = df[const.LABEL_BINARY_COL].value_counts(normalize=True)
        # discretecount = df[const.LABEL_DISCRETE_COL].value_counts(normalize=True)
        #
        # bin_fall = bincount.loc[0.0]
        # bin_raise = bincount.loc[1.0]
        #
        # dis_fall = discretecount.loc[0.0]
        # dis_keep = discretecount.loc[1.0]
        # dis_raise = discretecount.loc[2.0]

        style.use('ggplot')
        df[const.ADJUSTED_CLOSE_COL].plot(kind='line', x_compat=True, label='Cena zamknięcia')

        plt.title(sym)
        plt.ylabel('Cena zamknięcia (USD)')
        plt.xlabel('Data')
        plt.legend()
        # plt.show()
        plt.savefig('{}/{}.pdf'.format(COMPANY_FILES_PATH, sym + '_adj_close'),
                    format='pdf', dpi=1000)
        plt.savefig('{}/{}.png'.format(COMPANY_FILES_PATH, sym + '_adj_close'))
        plt.close()

        df[const.LABEL_BINARY_COL].plot(kind='hist', xticks=[0, 2])
        plt.xticks([0, 2], ['Spadek', 'Wzrost'])
        plt.title(sym)
        plt.xlabel('Klasa')
        plt.ylabel('Liczba próbek')
        plt.savefig('{}/{}.pdf'.format(COMPANY_FILES_PATH, sym + '_binary_hist'),
                    format='pdf', dpi=1000)
        plt.savefig('{}/{}.png'.format(COMPANY_FILES_PATH, sym + '_binary_hist'))
        plt.close()

        df[const.LABEL_DISCRETE_COL].plot(kind='hist', xticks=[0, 1, 2])
        plt.xticks([0, 1, 2], ['Spadek', 'Utrzymanie', 'Wzrost'])
        plt.title(sym)
        plt.xlabel('Klasa')
        plt.ylabel('Liczba próbek')
        plt.savefig('{}/{}.pdf'.format(COMPANY_FILES_PATH, sym + '_discrete_hist'),
                    format='pdf', dpi=1000)
        plt.savefig('{}/{}.png'.format(COMPANY_FILES_PATH, sym + '_discrete_hist'))
        plt.close()


if __name__ == '__main__':
    plot_close_line_and_label_histograms()
    compare_adj_one_plot()
    print('FINISHED')
