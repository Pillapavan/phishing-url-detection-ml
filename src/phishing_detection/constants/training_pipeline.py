# Training Pipeline
PIPELINE_NAME = "phishing_detection"
ARTIFACT_DIR = "artifacts"


URL_FEATURE_COLUMNS = [
    "url_len",
    "dom_len",
    "is_ip",
    "tld_len",
    "subdom_cnt",
    "letter_cnt",
    "digit_cnt",
    "special_cnt",
    "eq_cnt",
    "qm_cnt",
    "amp_cnt",
    "dot_cnt",
    "dash_cnt",
    "under_cnt",
    "letter_ratio",
    "digit_ratio",
    "spec_ratio",
    "is_https",
    "slash_cnt",
    "entropy",
    "path_len",
    "query_len"
]
TARGET_COLUMN = "label"

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
DATA_VALIDATION_MISSING_VALUE_THRESHOLD = 0.05


DATA_ANALYSIS_DIR_NAME = "data_analysis"
DATA_ANALYSIS_REPORT_FILE_NAME = "analysis_report.yaml"
DATA_ANALYSIS_IMBALANCE_THRESHOLD = 0.40


# Data Transformation
DATA_TRANSFORMATION_DIR_NAME = "data_transformation"

DATA_TRANSFORMATION_TRANSFORMED_DIR = "transformed"

DATA_TRANSFORMATION_TRAIN_FILE_NAME = "train.npy"
DATA_TRANSFORMATION_TEST_FILE_NAME = "test.npy"

DATA_TRANSFORMATION_PROCESSOR_FILE_NAME = "processor.pkl"

# Model training
MODEL_TRAINER_DIR_NAME = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME = "model.pkl"


# Model Evaluation
MODEL_EVALUATION_DIR_NAME = "model_evaluation"

MODEL_EVALUATION_REPORT_FILE_NAME = "evaluation_report.yaml"

MODEL_EVALUATION_CONFUSION_MATRIX_FILE_NAME = (
    "confusion_matrix.png"
)

MODEL_EVALUATION_ROC_CURVE_FILE_NAME = (
    "roc_curve.png"
)