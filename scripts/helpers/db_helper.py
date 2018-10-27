import pymongo

DB = "ai-broker-stock"


def create_db_connection():
    mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = mongo_client[DB]
    return db
