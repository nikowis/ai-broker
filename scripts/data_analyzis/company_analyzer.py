import matplotlib.pyplot as plt
from matplotlib import style

import csv_importer
import stock_constants as const


CSV_FILES_DIR = './../../target/data'

FORECAST_COUNT_LABEL = 'Liczba prognoz'
VALUE_CHANGE_LABEL = 'Zmiana wartości'

RATE_CHANGE_LABEL = 'Zmiana kursu'
RISE_LABEL = 'wzrost'
IDLE_LABEL = 'utrzymanie'
FALL_LABEL = 'spadek'


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
    company_info_path = './../../target/company_info'
    if not os.path.exists(company_info_path):
        os.makedirs(company_info_path)
    plt.savefig('{}/{}.png'.format(company_info_path, symbol))

    # plt.show()
    plt.close()

def main():
    symbols = ['USLM']
    df_list, _ = csv_importer.import_data_from_files(symbols, CSV_FILES_DIR)

    balanced_syms = []

    for i in range(0, len(symbols)):
        sym = symbols[i]
        df = df_list[i]
        MIN_DATE = '1900-01-01'
        MAX_DATE = '2020-10-29'
        df = df[(df.index > MIN_DATE)]
        df = df[(df.index < MAX_DATE)]

        bincount = df[const.LABEL_BINARY_COL].value_counts(normalize=True)
        discretecount = df[const.LABEL_DISCRETE_COL].value_counts(normalize=True)

        bin_fall = bincount.loc[0.0]
        bin_raise = bincount.loc[1.0]

        dis_fall = discretecount.loc[0.0]
        dis_keep = discretecount.loc[1.0]
        dis_raise = discretecount.loc[2.0]

        style.use('ggplot')

        df[const.ADJUSTED_CLOSE_COL].plot(kind='line')
        plt.title('Close price')
        plt.ylabel('Price')
        plt.xlabel('Date')
        plt.show()
        plt.close()

        df[const.LABEL_BINARY_COL].plot(kind='hist', xticks=[-1, 1])
        plt.xticks([-1, 1], ['Fall', 'Rise'])
        plt.xlabel('Class')
        plt.ylabel('Freq')
        plt.show()
        plt.close()

    print(len(balanced_syms))
    print(balanced_syms)


if __name__ == '__main__':
    main()
    print('FINISHED')
