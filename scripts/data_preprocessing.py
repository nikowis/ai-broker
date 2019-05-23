import matplotlib.pyplot as plt
from matplotlib import style
import os
import db_access
import stock_constants as const
import pandas as pd


MIN_DATE = '2019-01-01'
MAX_DATE = '2020-10-29'
SELECTED_SYM = 'GOOGL'



def preprocess(df, standarize=True, difference_non_stationary=True):
    pass


if __name__ == '__main__':
    db_conn = db_access.create_db_connection(remote=False)
    df_list, sym_list = db_access.find_by_tickers_to_dateframe_parse_to_df_list(db_conn, [SELECTED_SYM],
                                                                                min_date=MIN_DATE, max_date=MAX_DATE)
    df = df_list[0]
    preprocess(df)
