import pandas as pd


def import_data_from_files(tickers, path):
    df_list = []
    sym_list = []
    for ticker in tickers:
        file = path + '/' + ticker + '.csv'
        df = pd.read_csv(file, index_col=0)
        df.index = pd.to_datetime(df.index)
        df_list.append(df)
        sym_list.append(ticker)
    # print('Retreived all data from files')
    return df_list, sym_list
