import pymongo

DB = "ai-broker"
STOCK_COLLECTION = "stock"


def create_db_connection():
    mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = mongo_client[DB]
    return db


def stock_collection(db):
    return db[STOCK_COLLECTION]
