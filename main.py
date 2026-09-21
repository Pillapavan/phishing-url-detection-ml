from phishing_detection.components.data_ingestion import DataIngestion
from phishing_detection.components.data_validation import DataValidation
from phishing_detection.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig
)


def main():

    # -------------------------
    # Training Pipeline Config
    # -------------------------
    training_pipeline_config = TrainingPipelineConfig()

    # -------------------------
    # Data Ingestion
    # -------------------------
    data_ingestion_config = DataIngestionConfig(
        training_pipeline_config
    )

    data_ingestion = DataIngestion(
        data_ingestion_config
    )

    data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

    print("\n========== DATA INGESTION ==========")
    print(data_ingestion_artifact)

    # -------------------------
    # Data Validation
    # -------------------------
    data_validation_config = DataValidationConfig(
        training_pipeline_config
    )

    data_validation = DataValidation(
        data_ingestion_artifact,
        data_validation_config
    )

    data_validation_artifact = (
        data_validation.initiate_data_validation()
    )

    print("\n========== DATA VALIDATION ==========")
    print(data_validation_artifact)


if __name__ == "__main__":
    main()