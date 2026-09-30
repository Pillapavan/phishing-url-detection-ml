import os
import sys
import pickle
import logging
import yaml
import numpy as np
import mlflow
import mlflow.sklearn

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)

from phishing_detection.entity.config_entity import (
    ModelEvaluationConfig
)

from phishing_detection.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact
)

from phishing_detection.exception.exception import CustomException


class ModelEvaluation:

    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
        model_evaluation_config: ModelEvaluationConfig
    ):

        self.data_transformation_artifact = (
            data_transformation_artifact
        )

        self.model_trainer_artifact = (
            model_trainer_artifact
        )

        self.model_evaluation_config = (
            model_evaluation_config
        )

    def load_model(self):

        with open(
            self.model_trainer_artifact.trained_model_file_path,
            "rb"
        ) as file:

            model = pickle.load(file)

        return model

    def load_test_data(self):

        test_array = np.load(
            self.data_transformation_artifact
            .transformed_test_file_path
        )

        X_test = test_array[:, :-1]

        y_test = test_array[:, -1]

        return X_test, y_test

    def predict(self, model, X_test):

        predictions = model.predict(X_test)

        return predictions

    def calculate_metrics(
    self,
    y_test,
    predictions,
    probabilities
    ):

        metrics = {

            "accuracy": accuracy_score(
                y_test,
                predictions
            ),

            "precision": precision_score(
                y_test,
                predictions
            ),

            "recall": recall_score(
                y_test,
                predictions
            ),

            "f1": f1_score(
                y_test,
                predictions
            )
        }

        if probabilities is not None:

            metrics["roc_auc"] = roc_auc_score(
                y_test,
                probabilities
            )

        return metrics

    def generate_classification_report(
    self,
    y_test,
    predictions
    ):

        report = classification_report(
            y_test,
            predictions,
            output_dict=True
        )

        return report

    def generate_confusion_matrix(
    self,
    y_test,
    predictions
    ):

        matrix = confusion_matrix(
            y_test,
            predictions
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
                display_labels=[
                "Phishing",
                "Legitimate"
            ]
        )

        display.plot()

        plt.title("Confusion Matrix")

        os.makedirs(
            os.path.dirname(
                self.model_evaluation_config
                .confusion_matrix_file_path
            ),
            exist_ok=True
        )

        plt.savefig(
            self.model_evaluation_config
            .confusion_matrix_file_path
        )

        plt.close()

        return matrix

    def generate_roc_curve(
    self,
    y_test,
    probabilities
    ):

        fpr, tpr, _ = roc_curve(
            y_test,
            probabilities
        )

        plt.figure()

        plt.plot(
            fpr,
            tpr,
            label="ROC Curve"
        )

        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--"
        )

        plt.xlabel("False Positive Rate")

        plt.ylabel("True Positive Rate")

        plt.title("ROC Curve")

        plt.legend()

        os.makedirs(
            os.path.dirname(
                self.model_evaluation_config
                .roc_curve_file_path
            ),
            exist_ok=True
        )

        plt.savefig(
            self.model_evaluation_config
            .roc_curve_file_path
        )

        plt.close()

    def save_evaluation_report(
        self,
        metrics,
        classification_report_data
      ):

        report = {
            "metrics": metrics,
            "classification_report":
            classification_report_data
        }

        os.makedirs(
            os.path.dirname(
                self.model_evaluation_config
                .evaluation_report_file_path
            ),
            exist_ok=True
        )

        with open(
            self.model_evaluation_config
            .evaluation_report_file_path,
            "w"
        ) as file:

            yaml.safe_dump(
                report,
                file
            )

    def initiate_model_evaluation(self):

        try:

            logging.info(
                "Starting model evaluation"
            )

            mlflow.set_experiment("phishing_detection")

            with mlflow.start_run(run_name="final_model_evaluation"):

                # 1. Load trained model
                model = self.load_model()

                # 2. Load untouched test data
                X_test, y_test = self.load_test_data()

                # 3. Predictions
                predictions = self.predict(
                    model,
                    X_test
                )

                # 4. Probabilities
                probabilities = None

                if hasattr(model, "predict_proba"):

                    probabilities = model.predict_proba(
                        X_test
                    )[:, 1]

                # 5. Metrics
                metrics = self.calculate_metrics(
                    y_test,
                    predictions,
                    probabilities
                )

                # 6. Classification report
                report = self.generate_classification_report(
                    y_test,
                    predictions
                )

                # 7. Confusion matrix
                self.generate_confusion_matrix(
                    y_test,
                    predictions
                )

                # 8. ROC curve
                if probabilities is not None:

                    self.generate_roc_curve(
                        y_test,
                        probabilities
                    )

                # 9. Save report
                self.save_evaluation_report(
                    metrics,
                    report
                )

                # MLflow metrics
                mlflow.log_metrics(metrics)

                # MLflow artifacts
                mlflow.log_artifact(
                    self.model_evaluation_config.evaluation_report_file_path
                )

                mlflow.log_artifact(
                    self.model_evaluation_config.confusion_matrix_file_path
                )

                mlflow.log_artifact(
                    self.model_evaluation_config.roc_curve_file_path
                )

                logging.info(
                    f"Model evaluation completed: {metrics}"
                )
                mlflow.sklearn.log_model(
                        model,
                        "model"
                    )

            return ModelEvaluationArtifact(
                model_evaluation_status=True,
                evaluation_report_file_path=(
                    self.model_evaluation_config
                    .evaluation_report_file_path
                ),
                confusion_matrix_file_path=(
                    self.model_evaluation_config
                    .confusion_matrix_file_path
                ),
                roc_curve_file_path=(
                    self.model_evaluation_config
                    .roc_curve_file_path
                )
            )

        except Exception as e:

            logging.error(
                "Error occurred during model evaluation"
            )

            raise CustomException(e, sys)