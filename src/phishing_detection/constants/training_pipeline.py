# Training Pipeline
PIPELINE_NAME = "phishing_detection"
ARTIFACT_DIR = "artifacts"

# Data Ingestion
DATA_INGESTION_DIR_NAME = "data_ingestion"

DATA_INGESTION_FEATURE_STORE_DIR = "feature_store"
DATA_INGESTION_INGESTED_DIR = "ingested"

FILE_NAME = "phisingData.csv"

TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"

DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO = 0.2

# MongoDB
DATA_INGESTION_DATABASE_NAME = "phishing_detection"
DATA_INGESTION_COLLECTION_NAME = "phishing_data"