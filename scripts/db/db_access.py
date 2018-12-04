import pymongo
import pandas as pd
import db.stock_constants as const

DB = "ai-broker"
STOCK_COLLECTION = "stock"
PROCESSED_STOCK_COLLECTION = "processed_stock"
LOCAL_URL="mongodb://localhost:27017/"
REMOTE_URL="mongodb://admin:<pswd>@ds125574.mlab.com:25574/ai-broker"

def create_db_connection(remote=False):
    if not remote:
        url = LOCAL_URL
    else:
        url = REMOTE_URL
    mongo_client = pymongo.MongoClient(url)
    db_conn = mongo_client[DB]
    return db_conn


def stock_collection(db_conn, processed=True):
    if processed:
        return db_conn[PROCESSED_STOCK_COLLECTION]
    else:
        return db_conn[STOCK_COLLECTION]


def find_one_by_ticker(db_conn, symbol, processed=True):
    return stock_collection(db_conn, processed).find_one({const.SYMBOL: symbol})


def find_one_by_ticker_dateframe(db_conn, symbol, processed=True):
    data = find_one_by_ticker(db_conn, symbol, processed)
    data.pop(const.ID, None)
    data.pop(const.SYMBOL, None)
    df = pd.DataFrame.from_dict(data, orient=const.INDEX)
    df = df.astype(float)
    return df
