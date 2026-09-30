import os
import sys
import pandas as pd
from dotenv import load_dotenv
from phishing_detection.exception.exception import CustomException
from phishing_detection.logger.logger import logging
import pymongo
import certifi

load_dotenv()
ca = certifi.where()
MONGO_DB_URL=os.getenv("MONGO_DB_URL")
# print(MONGO_DB_URL)

class NetworkDataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise CustomException(e,sys)

    def csv_to_json_convertor(self,file_path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True,inplace=True)
            records = data.to_dict(orient="records")
            return records
        except Exception as e:
            raise CustomException(e,sys)

    def insert_data_mongodb(self,records,database,collection):
        try:
            self.database=database
            self.collection=collection
            self.records=records
            logging.info("Data ingestion started")
            self.mongodb_client = pymongo.MongoClient(MONGO_DB_URL,tlsCAFile=ca)
            self.database = self.mongodb_client[self.database]
            self.collection = self.database[self.collection]
            self.collection.insert_many(self.records)
            logging.info("Data successfully inserted into MongoDB")
            return len(self.records)

        except Exception as e:
            logging.error("Failed to connect to MongoDB")
            raise CustomException(e,sys)


if __name__=='__main__':
    FILE_PATH="data/Dataset.csv"
    DATABASE = "phishing_detection"
    COLLECTION = "phishing_data"
    networkobj=NetworkDataExtract()
    records=networkobj.csv_to_json_convertor(file_path=FILE_PATH)
    print(records)
    no_of_records=networkobj.insert_data_mongodb(records,DATABASE,COLLECTION)
    print(no_of_records)
        