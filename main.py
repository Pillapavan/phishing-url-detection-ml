from phishing_detection.components.data_ingestion import DataIngestion
from phishing_detection.components.data_validation import DataValidation
from phishing_detection.components.data_analysis import DataAnalysis
from phishing_detection.components.data_transformation import DataTransformation
from phishing_detection.components.model_trainer import ModelTrainer
from phishing_detection.components.model_evaluation import ModelEvaluation


from phishing_detection.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataAnalysisConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig
)

import dagshub
from dotenv import load_dotenv
import os
import mlflow

load_dotenv()

dagshub.init(
    repo_owner=os.getenv("DAGSHUB_USERNAME"),
    repo_name="phishing-url-detection-ml",
    mlflow=True
)

mlflow.set_experiment("phishing_detection")

def main():


    training_pipeline_config = TrainingPipelineConfig()

    # ============================================================
    # DATA INGESTION
    # ============================================================

    data_ingestion_config = DataIngestionConfig(
        training_pipeline_config
    )

    data_ingestion = DataIngestion(
        data_ingestion_config
    )

    data_ingestion_artifact = (
        data_ingestion.initiate_data_ingestion()
    )

    print("\n========== DATA INGESTION ==========")
    print(data_ingestion_artifact)


    # ============================================================
    # DATA VALIDATION
    # ============================================================

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


    # ============================================================
    # DATA ANALYSIS
    # ============================================================

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


    # ============================================================
    # DATA TRANSFORMATION
    # ============================================================

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


    # ============================================================
    # MODEL TRAINING
    # ============================================================

    model_trainer_config = ModelTrainerConfig(
        training_pipeline_config
    )

    model_trainer = ModelTrainer(
        data_transformation_artifact,
        model_trainer_config
    )

    model_trainer_artifact = (
        model_trainer.initiate_model_training()
    )

    print("\n========== MODEL TRAINING ==========")
    print(model_trainer_artifact)


    # ============================================================
    # MODEL EVALUATION
    # ============================================================

    model_evaluation_config = ModelEvaluationConfig(
        training_pipeline_config
    )

    model_evaluation = ModelEvaluation(
        data_transformation_artifact,
        model_trainer_artifact,
        model_evaluation_config
    )

    model_evaluation_artifact = (
        model_evaluation.initiate_model_evaluation()
    )

    print("\n========== MODEL EVALUATION ==========")
    print(model_evaluation_artifact)


if __name__ == "__main__":
    main()