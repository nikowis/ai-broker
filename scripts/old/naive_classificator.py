import numpy as np

import db_access

PRED = 'Prediction'
LABEL = 'Label binary'


def accuracy(df):
    labels = df[[LABEL]].as_matrix().T
    predictions = df[[PRED]].as_matrix().T
    different = labels != predictions
    error = np.mean(different)
    return error


def main():
    ticker = 'ACUR'
    db_conn = db_access.create_db_connection(remote=False)
    df_list, sym_list = db_access.find_by_tickers_to_dateframe_parse_to_df_list(db_conn, [ticker])
    df = df_list[0]
    df = df[[LABEL]]
    df = df[:-1]
    df.reset_index(drop=True, inplace=True)
    df[PRED] = 0.0
    acc = accuracy(df)
    print('Predicted default accuracy ', acc)

    for last in range(1, 5):
        mean_of_last(df, last)
        acc = accuracy(df)
        print('Predicted of last, ', last, 'accuracy ', acc)


def mean_of_last(df, last):
    for i, row in df.iterrows():
        if i >= last:
            prevs = df[LABEL][i - last:i]
            prevmean = np.mean(prevs)
            if prevmean >= 0.5:
                newval = 1.0
            else:
                newval = 0
            df.at[i, PRED] = newval


if __name__ == '__main__':
    main()
