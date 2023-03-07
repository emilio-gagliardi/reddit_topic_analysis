import datetime
import os
import sqlite3

import dotenv
from loguru import logger
import json
import pandas as pd
import pymongo
from typing import List, Tuple, Any, Dict
from pathlib import Path


def get_project_dir(cwd: str, base_dir: str) -> Path:
    # Get the current working directory
    current_dir = cwd

    # Loop until we find the topmost directory
    while True:
        # Check if we've reached the root directory
        if current_dir == os.path.abspath(os.sep):
            raise ValueError("Could not find project directory")

        # Check if the current directory has the expected structure
        if os.path.basename(current_dir) == base_dir and \
                os.path.basename(os.path.dirname(current_dir)) == "src":
            return os.path.dirname(os.path.dirname(current_dir))

        # Move up to the parent directory
        current_dir = os.path.dirname(current_dir)


PROJECT_NAME = "reddit_topic_analysis"
PROJECT_DIR = get_project_dir(os.getcwd(), PROJECT_NAME)
DATA_DIR = os.path.join(PROJECT_DIR, "src", PROJECT_NAME, "data")
MODEL_DIR = os.path.join(PROJECT_DIR, "src", PROJECT_NAME, "model")
LOGS_DIR = os.path.join(PROJECT_DIR, "src", PROJECT_NAME, "logs")
CONFIG_DIR = os.path.join(PROJECT_DIR, "src", PROJECT_NAME, ".config")

# (TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, or CRITICAL)
# LOGURU_LEVEL = "INFO"
os.environ["LOGURU_LEVEL"] = "DEBUG"
log_dev_file = os.path.join(LOGS_DIR, 'development.log')
logger.add(open(log_dev_file, "a"),
           format="{time:YYYY-MM-DD at HH:mm:ss} {module}::{function} [{level}] {message}",
           level="DEBUG")
env_variables_path = os.path.join(CONFIG_DIR, 'environment.env')

logger.debug(f"project dir is {PROJECT_DIR}")
logger.debug(f"data dir is {DATA_DIR}")
logger.debug(f"logs dir is {LOGS_DIR}")
logger.debug(f"config dir is {CONFIG_DIR}")


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


class SQLite3Connection:
    """SQLite3 Connection"""

    def __init__(self, dbname=None):
        """
        Set the database name
        """

        if dbname is None:
            self.dbname = 'reddit_data.db'
        else:
            self.dbname = dbname
        self.connection = None

    def __enter__(self):
        self.connection = sqlite3.connect(self.dbname)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.connection.close()


def load_environment_variables(env_file: str) -> None:
    if os.path.isfile(env_file):
        # Load the environment variables from the .env file
        dotenv.load_dotenv(dotenv_path=env_file)
        logger.info("Loading environment variables.")
    else:
        logger.warning("File not found. Could not load environment variables.")


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            return obj.decode('utf-8')
        return json.JSONEncoder.default(self, obj)


def dicts_to_json(data_list: List[Dict]) -> str:
    if len(data_list) > 0:
        try:
            json_data = []
            for item in data_list:
                json_item = {}
                for key, value in item.items():
                    if isinstance(value, bytes):
                        value = value.decode('utf-8')
                    json_item[key] = value
                json_data.append(json_item)
            return json.dumps(json_data, cls=CustomJSONEncoder)
        except TypeError as e:
            print(f"Error serializing data: {e}")
            print(f"Offending item: {e.args[0].split(' ')[3]}")
    else:
        print("No data to convert to JSON.")
        return None


def save(data,
         file_type: str = "documents",
         db_name: str = None,
         collection_name: str = None,
         db_table: str = None,
         col_names: List[str] = None) -> None:
    """
    based on the type of data, save it in the appropriate format
    or to the appropriate location. save list of dicts to a document collection,
    save a list of tuples to a database table, and save JSON objects to a file.
    Args:
        data:
        file_type: the type of save you want to do. Can be "documents", "tables", or "json"
        db_name: name of the database
        collection_name: name of the collection
        db_table: name of the table
        col_names: list of column names

    Returns:
        None
    """
    if file_type == "documents":
        if isinstance(data, list):
            save_as_documents(data, db_name, collection_name)
        else:
            logger.error("Data is not a list of dicts. Could not save.")
    elif file_type == "tables":
        if isinstance(data, list):
            save_as_table(data, db_name, db_table, col_names)
        else:
            logger.error("Data is not a list of tuples. Could not save.")
    elif file_type == "json":
        if isinstance(data, str):
            save_as_json(data)
        else:
            logger.error("Data is not a JSON serialized string. Could not save.")
    else:
        logger.error("File type not recognized. Could not save.")


def save_as_documents(data: List[Dict], db_name: str, collection_name: str) -> None:
    """
    Save a list of dicts as documents in a MongoDB collection.
    Args:
        data: a list of dicts to save
        db_name: name of the database
        collection_name: the collection name

    Returns:
        None
    """
    if len(data) > 0:
        json_data = dicts_to_json(data)
        try:
            with MongoDBConnection() as datastore:
                db = datastore.connection[db_name]
                collection = db[collection_name]
                logger.debug(collection)
                json_obj = json.loads(json_data)
                logger.debug(json_obj)
                collection.insert_many(json_obj)
        except Exception as e:
            logger.error(e)

        logger.info(f"Saved {len(data)} documents to {collection_name}.")
    else:
        logger.warning("No data to save.")


def save_as_table(data: List[Tuple[Any]], db_name: str, db_table: str, col_names: List[str]) -> None:
    """
    Save a list of tuples to a table in a database.
    Args:
        data: a list of tuples to save
        db_name: name of the database
        db_table: name of the table
        col_names: list of column names for INSERT statement

    Returns:
        None
    """
    if len(data) > 0:
        try:
            with SQLite3Connection(db_name) as conn:
                cursor = conn.connection.cursor()
                placeholders = ', '.join('?' * len(col_names))
                insert_sql = f"INSERT INTO {db_table} ({', '.join(col_names)}) VALUES ({placeholders})"
                cursor.executemany(insert_sql, data)
                conn.connection.commit()
        except Exception as e:
            logger.error(e)

        logger.info(f"Saved {len(data)} rows to {db_table}.")
    else:
        logger.warning("No data to save.")


def save_as_json(data: str) -> None:
    """
    Save a JSON serialized string to a file.
    Args:
        data: json formatted string

    Returns:
        None
    """
    if data is not None:
        # Get the current date and time
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Save the data to a file
        json_file = os.path.join(DATA_DIR, f"submissions_{now}.json")
        with open(json_file, "w") as f:
            f.write(data)
        logger.info(f"Saved data to {json_file}.")
    else:
        logger.warning("No data to save.")


def load_documents(db_name, collection_name, query=None, limit=10):

    try:
        with MongoDBConnection() as mongo:
            db = mongo.connection[db_name]
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
    print("running main")
    load_environment_variables(env_variables_path)

    menu_options = ["get reddit submissions", "save raw submissions", "process raw submissions",
                    "save cleaned submissions", "get cleaned submissions", "make BoW matrix",
                    "make tfidf matrix", "make similarity matrix", "save similarity matrix", "exit"]

    model_path = os.path.join(MODEL_DIR, "rta_nlp")
    if not os.path.isdir(model_path):
        logger.debug(f"Model not found at {model_path}. Creating new model.")
        reddit_topic_analysis.data.data_processing.model_setup(model_path)

    nlp = reddit_topic_analysis.data.data_processing.load_spacy_language_model(model_path)
    spacy_stopwords = nlp.Defaults.stop_words

    while True:
        for option_number, option_name in enumerate(menu_options, 1):
            print(f"{option_number}. {option_name}")
        selection = int(input("Type your menu selection: "))

        if selection not in range(len(menu_options) + 1):
            print("Incorrect option, please try again...")
            continue
        print(f"you selected {selection}")
        match str(selection):
            case '1':
                print("get reddit submissions")
                creds = reddit_topic_analysis.data.data_processing.get_reddit_credentials()
                reddit_conn = reddit_topic_analysis.data.data_processing.connect_to_reddit_with_oauth(creds[0],
                                                                                                      creds[1])
                subreddit_conn = reddit_topic_analysis.data.data_processing.get_one_subreddit(reddit_conn, "Intune")
                submissions = reddit_topic_analysis.data.data_processing.extract_submission_info(subreddit_conn,
                                                                                                 "hot", 3)
                print(submissions)
            case '2':
                print("=== save raw submissions ===")
                creds = reddit_topic_analysis.data.data_processing.get_reddit_credentials()
                reddit_conn = reddit_topic_analysis.data.data_processing.connect_to_reddit_with_oauth(creds[0],
                                                                                                      creds[1])
                sub_name = (input("Subreddit name: "))
                subreddit_conn = reddit_topic_analysis.data.data_processing.get_one_subreddit(reddit_conn, sub_name)
                num_subs = int(input("Number of submissions: "))
                submissions = reddit_topic_analysis.data.data_processing.extract_submission_info(subreddit_conn,
                                                                                                 "new", num_subs)
                save(data=submissions, file_type="documents",
                     db_name="reddit_tier_3", collection_name="reddit_submissions")
            case '3':
                print("=== process raw submissions ===")
                # get raw submissions from MongoDB
                submissions = load_documents(db_name="reddit_tier_3", collection_name="reddit_submissions",
                                             query=None, limit=10)
                json_payload = dicts_to_json(submissions)
                logger.debug(json_payload)
                # save to MongoDB
                # save_to_mongo_one(json_payload, "reddit_submissions")
            case '4':
                print("=== save cleaned submissions ===")
            case '5':
                print("=== get cleaned submissions ===")
            case '6':
                print("=== make BoW matrix ===")
            case '7':
                print("=== make tfidf matrix ===")
            case '8':
                print("=== make similarity matrix ===")
            case '9':
                print("=== save similarity matrix ===")
            case '10':
                print("exit")
                break


if __name__ == "__main__":
    import reddit_topic_analysis.data.data_processing

    main()
