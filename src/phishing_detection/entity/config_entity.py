import os
from datetime import datetime

from phishing_detection.constants import training_pipeline

class TrainingPipelineConfig:

    def __init__(self,timestamp=None):
        if timestamp is None:
            timestamp = datetime.now()
        timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")

        self.pipeline_name = training_pipeline.PIPELINE_NAME
        self.artifact_name = training_pipeline.ARTIFACT_DIR

        self.artifact_dir = os.path.join(self.artifact_name,timestamp)

        self.model_dir = os.path.join(
            "final_model"
        )

        self.timestamp: str = timestamp

class DataIngestionConfig:

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):

        self.data_ingestion_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_INGESTION_DIR_NAME
        )
        self.feature_store_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR,
            training_pipeline.FILE_NAME
        )
        self.training_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.TRAIN_FILE_NAME
        )
        self.testing_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.TEST_FILE_NAME
        )
        self.train_test_split_ratio: float = (
            training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        )
        self.collection_name: str = (
            training_pipeline.DATA_INGESTION_COLLECTION_NAME
        )
        self.database_name: str = (
            training_pipeline.DATA_INGESTION_DATABASE_NAME
        )
        

class DataValidationConfig:
    def __init__(
        self,
        training_pipeline_config: TrainingPipelineConfig
    ):
        self.data_validation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_VALIDATION_DIR_NAME
        )
        self.valid_data_dir: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_VALID_DIR
        )
        self.invalid_data_dir: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_INVALID_DIR
        )
        self.drift_report_dir: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR
        )
        self.valid_train_file_path: str = os.path.join(
            self.valid_data_dir,
            training_pipeline.DATA_VALIDATION_VALID_TRAIN_FILE_NAME
        )
        self.valid_test_file_path: str = os.path.join(
            self.valid_data_dir,
            training_pipeline.DATA_VALIDATION_VALID_TEST_FILE_NAME
        )
        self.invalid_train_file_path: str = os.path.join(
            self.invalid_data_dir,
            training_pipeline.DATA_VALIDATION_INVALID_TRAIN_FILE_NAME
        )
        self.invalid_test_file_path: str = os.path.join(
            self.invalid_data_dir,
            training_pipeline.DATA_VALIDATION_INVALID_TEST_FILE_NAME
        )
        self.drift_report_file_path: str = os.path.join(
            self.drift_report_dir,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME
        )
        self.schema_file_path: str = os.path.join(
            "config",
            training_pipeline.DATA_VALIDATION_SCHEMA_FILE_NAME
        )

        self.missing_value_threshold = (
            training_pipeline.DATA_VALIDATION_MISSING_VALUE_THRESHOLD
        )

class DataAnalysisConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):

        self.data_analysis_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_ANALYSIS_DIR_NAME
        )

        self.analysis_report_file_path = os.path.join(
            self.data_analysis_dir,
            training_pipeline.DATA_ANALYSIS_REPORT_FILE_NAME
        )

        self.imbalance_threshold = (
            training_pipeline.DATA_ANALYSIS_IMBALANCE_THRESHOLD
        )


class DataTransformationConfig:

    def __init__(
        self,
        training_pipeline_config: TrainingPipelineConfig
    ):

        self.data_transformation_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_TRANSFORMATION_DIR_NAME
        )

        self.transformed_dir = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DIR
        )

        self.transformed_train_file_path = os.path.join(
            self.transformed_dir,
            training_pipeline.DATA_TRANSFORMATION_TRAIN_FILE_NAME
        )

        self.transformed_test_file_path = os.path.join(
            self.transformed_dir,
            training_pipeline.DATA_TRANSFORMATION_TEST_FILE_NAME
        )

        self.processor_file_path = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_PROCESSOR_FILE_NAME
        )


class ModelTrainerConfig:

    def __init__(
        self,
        training_pipeline_config: TrainingPipelineConfig
    ):

        self.model_trainer_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.MODEL_TRAINER_DIR_NAME
        )

        self.trained_model_dir = os.path.join(
            self.model_trainer_dir,
            training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR
        )

        self.trained_model_file_path = os.path.join(
            self.trained_model_dir,
            training_pipeline.MODEL_TRAINER_TRAINED_MODEL_NAME
        )


class ModelEvaluationConfig:

    def __init__(
        self,
        training_pipeline_config: TrainingPipelineConfig
    ):

        self.model_evaluation_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.MODEL_EVALUATION_DIR_NAME
        )

        self.evaluation_report_file_path = os.path.join(
            self.model_evaluation_dir,
            training_pipeline.MODEL_EVALUATION_REPORT_FILE_NAME
        )

        self.confusion_matrix_file_path = os.path.join(
            self.model_evaluation_dir,
            training_pipeline.MODEL_EVALUATION_CONFUSION_MATRIX_FILE_NAME
        )

        self.roc_curve_file_path = os.path.join(
            self.model_evaluation_dir,
            training_pipeline.MODEL_EVALUATION_ROC_CURVE_FILE_NAME
        )