import matplotlib.pyplot as plt
from matplotlib import style

import csv_importer
import stock_constants as const


CSV_FILES_DIR = const.TARGET_DIR + '/data'


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
