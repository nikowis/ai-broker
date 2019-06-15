import os

import csv_importer
import stock_constants as const

MIN_DATE = '2019-05-01'
MAX_DATE = '2020-05-10'
SELECTED_SYM = 'GOOGL'
IMG_PATH = const.TARGET_DIR + '/documentation_plots_and_images'
if not os.path.exists(IMG_PATH):
    os.makedirs(IMG_PATH)


def label_creation_steps_to_csv(df, filename):
    df2 = df[[const.ADJUSTED_CLOSE_COL, const.LABEL_COL, const.DAILY_PCT_CHANGE_COL, const.LABEL_DISCRETE_COL,
              const.LABEL_BINARY_COL]]
    print(df2.tail())

    df2.to_csv(path_or_buf='{}/{}.csv'.format(IMG_PATH, filename))


if __name__ == '__main__':
    df_list, _ = csv_importer.import_data_from_files([SELECTED_SYM])
    df = df_list[0]
    df = df_list[0]
    label_creation_steps_to_csv(df, 'label_creation')
