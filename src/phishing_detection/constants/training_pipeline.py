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


# Data Validation
# Data Validation

DATA_VALIDATION_DIR_NAME = "data_validation"
DATA_VALIDATION_VALID_DIR = "validated"
DATA_VALIDATION_INVALID_DIR = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR = "drift_report"
DATA_VALIDATION_VALID_TRAIN_FILE_NAME = "valid_train.csv"
DATA_VALIDATION_VALID_TEST_FILE_NAME = "valid_test.csv"
DATA_VALIDATION_INVALID_TRAIN_FILE_NAME = "invalid_train.csv"
DATA_VALIDATION_INVALID_TEST_FILE_NAME = "invalid_test.csv"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME = "drift_report.yaml"
DATA_VALIDATION_SCHEMA_FILE_NAME = "schema.yaml"