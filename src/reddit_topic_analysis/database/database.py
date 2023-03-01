import os
import sys
from loguru import logger
import pymongo
import json


class MongoDBConnection:
    """MongoDB Connection"""

    def __init__(self, host='127.0.0.1', port=27017):
        """
        be sure to use the ip address not name for local windows
        CAUTION: Don't do this in production!!!
        """

        self.host = host
        self.port = port
        self.connection = None

    def __enter__(self):
        self.connection = pymongo.MongoClient(self.host, self.port)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.connection.close()


def print_mdb_collection(collection_name):
    for doc in collection_name.find():
        print(doc)


def save_to_mongo_many(json_data, collection_name):
    """Save a JSON object to MongoDB
    :param json_data: JSON object
    :param collection_name: Name of the collection to save to
    :returns: None"""
    try:
        with MongoDBConnection() as mongo:
            db = mongo.connection.reddit_tier_3
            collection = db[collection_name]
            logger.debug(collection)
            json_obj = json.loads(json_data)
            collection.insert_many(json_obj)
    except Exception as e:
        logger.error(e)
    return None


def save_to_mongo_one(json_data: str, collection_name: str) -> None:
    """Save a JSON object to MongoDB
    :param json_data: JSON formatted string
    :param collection_name: Name of the collection to save to
    :returns: None"""
    try:
        with MongoDBConnection() as mongo:
            db = mongo.connection.reddit_tier_3
            collection = db[collection_name]
            logger.debug(vars(collection))
            json_obj = json.loads(json_data)
            logger.debug(json_obj)
            if isinstance(json_obj, list):
                for obj in json_obj:
                    logger.debug(obj)
                    collection.insert_one(obj)
            else:
                logger.debug(json_obj)
                collection.insert_one(json_obj)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decoding error: {e}")
    except Exception as e:
        logger.error(f"Error saving data to MongoDB: {e}")
    return None


def load_from_mongo(collection_name, query=None, limit=10):
    """with the MongoDBConnection() as mongo: connection, query MongoDB
    connect to the collection and retrieve all documents
    :param collection_name: Name of the collection to load from
    :param query: Query to filter the collection
    :param limit: Number of documents to return
    :returns: List of documents"""
    try:
        with MongoDBConnection() as mongo:
            db = mongo.connection.reddit_tier_3
            collection = db[collection_name]
            if query is None:
                cursor = collection.find().limit(limit)
            else:
                logger.debug(f"Query: {query}")
                cursor = collection.find(query).limit(limit)

            data = [doc for doc in cursor]
            df = pd.DataFrame(data)
            logger.debug(df.head(3))
            return df
    except Exception as e:
        logger.error(f"Couldn't connect to mongoDB{e}")
        return None


def main():
    mongo = MongoDBConnection()
    with mongo:
        db = mongo.connection.test
        print(db)


if __name__ == "__main__":
    main()
