import os
import sys
import certifi
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
load_dotenv()


from phishing_detection.entity.artifact_entity import DataIngestionArtifact
from phishing_detection.entity.config_entity import (TrainingPipelineConfig,DataIngestionConfig)
from phishing_detection.constants import training_pipeline

from phishing_detection.exception.exception import CustomException
from phishing_detection.logger.logger import logging


class DataIngestion:

    def __init__(self,data_ingestion_config:DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config

    # Reading data from Mongodb
    def export_collection_as_dataframe(self) -> pd.DataFrame:
        try:
            mongo_db_url = os.getenv("MONGO_DB_URL")

            if not mongo_db_url:
                raise Exception("MONGO_DB_URL is not set")

            ca = certifi.where()

            client = MongoClient(
                mongo_db_url,
                tlsCAFile=ca
            )

            database = client[
                self.data_ingestion_config.database_name
            ]

            collection = database[
                self.data_ingestion_config.collection_name
            ]

            data = collection.find()

            data = list(data)

            if not data:
                raise Exception("No data found in MongoDB collection")

            dataframe = pd.DataFrame(data)

            if "_id" in dataframe.columns:
                dataframe.drop("_id", axis=1, inplace=True)

            return dataframe    
        except Exception as e:
            raise CustomException(e,sys)

    def save_data_to_feature_store(self,dataframe:pd.DataFrame) -> None:
        try:
           feature_store_dir =os.path.dirname(
            self.data_ingestion_config.feature_store_file_path
            )

           os.makedirs(feature_store_dir,exist_ok=True)

           dataframe.to_csv(self.data_ingestion_config.feature_store_file_path,index=False,header=True)

           logging.info("Data saved successfully in feature store")

        except Exception as e:
            raise CustomException(e,sys)


    def split_data_as_train_test(self,dataframe: pd.DataFrame) -> None:
        try:
            train_set,test_set = train_test_split(dataframe,test_size=self.data_ingestion_config.train_test_split_ratio,random_state=42)

            ingested_dir = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(
                ingested_dir,
                exist_ok=True
            )
            train_set.to_csv(
                self.data_ingestion_config.training_file_path,
                index=False
            )

            test_set.to_csv(
                self.data_ingestion_config.testing_file_path,
                index=False
            )

            logging.info(
                "Train and test data split successfully"
            )
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:

        try:

            logging.info("Starting data ingestion")

            dataframe = self.export_collection_as_dataframe()

            logging.info(
                f"Data retrieved from MongoDB. Shape: {dataframe.shape}"
            )

            self.save_data_to_feature_store(dataframe)

            self.split_data_as_train_test(dataframe)

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )

            logging.info(
                "Data ingestion completed successfully"
            )

            return data_ingestion_artifact

        except Exception as e:
            raise CustomException(e, sys)    