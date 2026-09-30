from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:

    trained_file_path: str
    test_file_path: str


@dataclass
class DataValidationArtifact:
    validation_status: bool

    valid_train_file_path: str
    valid_test_file_path: str

    invalid_train_file_path: str
    invalid_test_file_path: str

    drift_report_file_path: str


@dataclass
class DataAnalysisArtifact:
    is_imbalanced: bool
    analysis_report_file_path: str

@dataclass
class DataTransformationArtifact:

    transformed_train_file_path: str
    transformed_test_file_path: str
    processor_file_path: str

@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    best_model_name: str
    best_model_parameters: dict
    best_cv_score: float
    train_metric_artifact: dict


@dataclass
class ModelEvaluationArtifact:

    model_evaluation_status: bool
    evaluation_report_file_path: str
    confusion_matrix_file_path: str
    roc_curve_file_path: str