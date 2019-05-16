import db_access

MIN_DATE = '1900-01-01'
MAX_DATE = '2020-10-29'
SELECTED_SYM = 'GOOGL'


def main():
    db_conn = db_access.create_db_connection(remote=False)
    df_list, sym_list = db_access.find_by_tickers_to_dateframe_parse_to_df_list(db_conn, [SELECTED_SYM],
                                                                                min_date=MIN_DATE, max_date=MAX_DATE)


if __name__ == '__main__':
    main()
