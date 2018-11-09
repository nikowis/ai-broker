import pymongo

DB = "ai-broker"
STOCK_COLLECTION = "stock"
PROCESSED_STOCK_COLLECTION = "processed_stock"

def create_db_connection():
    mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = mongo_client[DB]
    return db


def stock_collection(db, processed=False):
    if processed:
        return db[PROCESSED_STOCK_COLLECTION]
    else:
        return db[STOCK_COLLECTION]
