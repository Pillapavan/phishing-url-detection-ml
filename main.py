from phishing_detection.components.data_ingestion import DataIngestion
from phishing_detection.components.data_validation import DataValidation
from phishing_detection.components.data_analysis import DataAnalysis
from phishing_detection.components.data_transformation import DataTransformation
from phishing_detection.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataAnalysisConfig,
    DataTransformationConfig
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


     # ================= DATA ANALYSIS =================

    data_analysis_config = DataAnalysisConfig(
        training_pipeline_config
    )

    data_analysis = DataAnalysis(
        data_validation_artifact,
        data_analysis_config
    )

    data_analysis_artifact = (
        data_analysis.initiate_data_analysis()
    )

    print("\n========== DATA ANALYSIS ==========")
    print(data_analysis_artifact)


    # ================= DATA TRANSFORMATION =================

    data_transformation_config = DataTransformationConfig(
        training_pipeline_config
    )

    data_transformation = DataTransformation(
        data_validation_artifact,
        data_analysis_artifact,
        data_transformation_config
    )

    data_transformation_artifact = (
        data_transformation.initiate_data_transformation()
    )

    print("\n========== DATA TRANSFORMATION ==========")
    print(data_transformation_artifact)


if __name__ == "__main__":
    main()