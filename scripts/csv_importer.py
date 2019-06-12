import pandas as pd
import stock_constants as const

CSV_FILES_DIR = const.TARGET_DIR + '/data'

def import_data_from_files(tickers, path=CSV_FILES_DIR):
    df_list = []
    for ticker in tickers:
        file = path + '/' + ticker + '.csv'
        df = pd.read_csv(file, index_col=0)
        df_list.append(df)
    print('Retreived all data from files')
    return df_list
