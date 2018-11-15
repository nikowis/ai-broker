from db import db_access
from helpers import data_helper
from neural_networks import nn_runner


def main():
    ticker = 'GOOGL'
    db_conn = db_access.create_db_connection()
    df = db_access.find_one_by_ticker_dateframe(db_conn, ticker)

    history_days = 2
    df, x_standarized, x_train, x_test, y_train_binary, y_test_binary = data_helper.extract_data(df, 1, history_days)
    nn_runner.run(df, x_standarized, x_train, x_test, y_train_binary, y_test_binary, history_days=history_days)


if __name__ == '__main__':
    main()
