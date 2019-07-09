import pandas as pd
import pymongo

import stock_constants as const

DB = "ai-broker"
STOCK_COLLECTION = "stock"
PROCESSED_STOCK_COLLECTION = "processed_stock"
LOCAL_URL = "mongodb://localhost:27017/"
REMOTE_URL = "mongodb://admin:<pswd>@ds125574.mlab.com:25574/ai-broker"

MIN_DATE = '1900-01-01'
MAX_DATE = '2020-10-29'


def create_db_connection(remote=False, db_name=DB):
    if not remote:
        url = LOCAL_URL
    else:
        url = REMOTE_URL
    mongo_client = pymongo.MongoClient(url)
    db_conn = mongo_client[db_name]
    return db_conn


def stock_collection(db_conn, processed=True):
    if processed:
        return db_conn[PROCESSED_STOCK_COLLECTION]
    else:
        return db_conn[STOCK_COLLECTION]


def find_by_tickers_to_dateframe_parse_to_df_list(db_conn, symbol_list, processed=True, min_date=MIN_DATE, max_date=MAX_DATE):
    data = stock_collection(db_conn, processed).find({const.SYMBOL_KEY: {"$in": symbol_list}})
    df_list = []
    symbol_output_list = []
    for document in data:
        symbol_output_list.append(document[const.SYMBOL_KEY])
        document.pop(const.ID, None)
        document.pop(const.SYMBOL_KEY, None)
        df = pd.DataFrame.from_dict(document, orient=const.INDEX)
        df = df.astype(float)
        df = df[(df.index > min_date)]
        df = df[(df.index < max_date)]
        df_list.append(df)
    if len(df_list) == 0:
        raise Exception('No data with any ticker of ' + str(symbol_list) + ' was found.')
    return df_list, symbol_output_list


if __name__ == "__main__":
    db_conn = create_db_connection()
    raw_collection = stock_collection(db_conn, False)
    data = raw_collection.find()
    symbol_list = []
    for document in data:
        sym = document[const.SYMBOL_KEY]
        symbol_list.append(sym)

    print(symbol_list)

