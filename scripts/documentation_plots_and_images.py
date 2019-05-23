import os

import db_access
import stock_constants as const

MIN_DATE = '2019-05-01'
MAX_DATE = '2020-05-10'
SELECTED_SYM = 'GOOGL'
IMG_PATH = './../target/documentation_plots_and_images'
if not os.path.exists(IMG_PATH):
    os.makedirs(IMG_PATH)


def label_creation_steps_to_csv(df, filename):
    df2 = df[[const.ADJUSTED_CLOSE_COL, const.LABEL_COL, const.DAILY_PCT_CHANGE_COL, const.LABEL_DISCRETE_COL, const.LABEL_BINARY_COL]]
    print(df2.tail())

    df2.to_csv(path_or_buf='{}/{}.csv'.format(IMG_PATH, filename))


if __name__ == '__main__':
    db_conn = db_access.create_db_connection(remote=False)
    df_list, sym_list = db_access.find_by_tickers_to_dateframe_parse_to_df_list(db_conn, [SELECTED_SYM],
                                                                                min_date=MIN_DATE, max_date=MAX_DATE)
    df = df_list[0]
    label_creation_steps_to_csv(df, 'label_creation')
