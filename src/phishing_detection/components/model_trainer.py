import os
import pickle
import numpy as np
import sys
import mlflow

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from scipy.stats import randint, uniform

from phishing_detection.exception.exception import CustomException
from phishing_detection.logger.logger import logging

from phishing_detection.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact
)

from phishing_detection.entity.config_entity import (
    ModelTrainerConfig
)


class ModelTrainer:

    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig
    ):

        self.data_transformation_artifact = (
            data_transformation_artifact
        )

        self.model_trainer_config = model_trainer_config

    # ---------------------------------------------------------
    # GET MODELS
    # ---------------------------------------------------------

    def get_models(self):

        models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                random_state=42
            ),

            "Random Forest": RandomForestClassifier(
                n_estimators=200,
                random_state=42
            ),

            "Gradient Boosting": GradientBoostingClassifier(
                random_state=42
            )
        }

        return models

    # ---------------------------------------------------------
    # HYPERPARAMETER DISTRIBUTIONS
    # ---------------------------------------------------------

    def get_parameter_distributions(self):

        parameter_distributions = {

            "Logistic Regression": {
                "C": uniform(0.01, 10)
            },

            "Random Forest": {
                "n_estimators": randint(100, 400),
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": randint(2, 10),
                "min_samples_leaf": randint(1, 5)
            },

            "Gradient Boosting": {
                "n_estimators": randint(50, 250),
                "learning_rate": uniform(0.01, 0.2),
                "max_depth": randint(2, 6)
            }
        }

        return parameter_distributions

    # ---------------------------------------------------------
    # CROSS VALIDATION STRATEGY
    # ---------------------------------------------------------

    def get_cv_strategy(self):

        return StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )

    # ---------------------------------------------------------
    # HYPERPARAMETER TUNING
    # ---------------------------------------------------------

    def tune_models(self, X_train, y_train):

        models = self.get_models()
        parameter_distributions = (
            self.get_parameter_distributions()
        )
        cv = self.get_cv_strategy()

        tuned_models = {}
        cv_scores = {}
        best_parameters = {}

        for model_name, model in models.items():

            logging.info(
                f"Starting hyperparameter tuning for {model_name}"
            )

            # Each model tuning becomes a child run
            with mlflow.start_run(
                run_name=f"{model_name}_tuning",
                nested=True
            ):

                search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=parameter_distributions[
                        model_name
                    ],
                    n_iter=10,
                    scoring="f1",
                    cv=cv,
                    random_state=42,
                    n_jobs=-1
                )

                search.fit(X_train, y_train)

                best_model = search.best_estimator_
                best_score = search.best_score_
                best_params = search.best_params_

                # ---------------------------------------------
                # MLflow logging for tuning run
                # ---------------------------------------------

                mlflow.log_param(
                    "model_name",
                    model_name
                )

                mlflow.log_params(best_params)

                mlflow.log_metric(
                    "cv_f1",
                    float(best_score)
                )

                tuned_models[model_name] = best_model

                cv_scores[model_name] = best_score

                best_parameters[model_name] = best_params

                logging.info(
                    f"{model_name} best CV F1: "
                    f"{search.best_score_}"
                )

                logging.info(
                    f"{model_name} best parameters: "
                    f"{search.best_params_}"
                )

        return (
            tuned_models,
            cv_scores,
            best_parameters
        )

    # ---------------------------------------------------------
    # SELECT BEST MODEL
    # ---------------------------------------------------------

    def select_best_model(
        self,
        tuned_models,
        cv_scores
    ):

        best_model_name = max(
            cv_scores,
            key=cv_scores.get
        )

        best_model = tuned_models[
            best_model_name
        ]

        logging.info(
            f"Best model selected: {best_model_name}"
        )

        logging.info(
            f"Best CV F1: "
            f"{cv_scores[best_model_name]}"
        )

        return (
            best_model_name,
            best_model
        )

    # ---------------------------------------------------------
    # TRAIN BEST MODEL
    # ---------------------------------------------------------

    def train_best_model(
        self,
        best_model,
        X_train,
        y_train
    ):

        logging.info(
            "Training selected model on complete training data"
        )

        best_model.fit(
            X_train,
            y_train
        )

        return best_model

    # ---------------------------------------------------------
    # EVALUATE MODEL
    # ---------------------------------------------------------

    def evaluate_model(
        self,
        model,
        X,
        y
    ):

        predictions = model.predict(X)

        metrics = {
            "accuracy": accuracy_score(
                y,
                predictions
            ),

            "precision": precision_score(
                y,
                predictions
            ),

            "recall": recall_score(
                y,
                predictions
            ),

            "f1_score": f1_score(
                y,
                predictions
            )
        }

        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = model.predict_proba(
                X
            )[:, 1]

            metrics["roc_auc"] = roc_auc_score(
                y,
                probabilities
            )

        return metrics

    # ---------------------------------------------------------
    # SAVE MODEL
    # ---------------------------------------------------------

    def save_model(self, model):

        os.makedirs(
            self.model_trainer_config.trained_model_dir,
            exist_ok=True
        )

        with open(
            self.model_trainer_config.trained_model_file_path,
            "wb"
        ) as file:

            pickle.dump(
                model,
                file
            )

        logging.info(
            f"Model saved at: "
            f"{self.model_trainer_config.trained_model_file_path}"
        )

    # ---------------------------------------------------------
    # INITIATE MODEL TRAINING
    # ---------------------------------------------------------

    def initiate_model_training(self):

        try:

            logging.info(
                "Starting model training pipeline"
            )

            # -------------------------------------------------
            # MLflow experiment
            # -------------------------------------------------

            mlflow.set_experiment(
                "phishing_detection"
            )

            # -------------------------------------------------
            # Parent MLflow run
            # -------------------------------------------------

            with mlflow.start_run(
                run_name="model_training"
            ):

                # ---------------------------------------------
                # Load transformed training data
                # ---------------------------------------------

                train_array = np.load(
                    self.data_transformation_artifact
                    .transformed_train_file_path
                )

                X_train = train_array[:, :-1]
                y_train = train_array[:, -1]

                logging.info(
                    f"Training data shape: "
                    f"{X_train.shape}"
                )

                # ---------------------------------------------
                # 1. Hyperparameter tuning
                # ---------------------------------------------

                (
                    tuned_models,
                    cv_scores,
                    best_parameters
                ) = self.tune_models(
                    X_train,
                    y_train
                )

                # ---------------------------------------------
                # 2. Select best model
                # ---------------------------------------------

                (
                    best_model_name,
                    best_model
                ) = self.select_best_model(
                    tuned_models,
                    cv_scores
                )

                # ---------------------------------------------
                # 3. Train best model on complete train data
                # ---------------------------------------------

                best_model = self.train_best_model(
                    best_model,
                    X_train,
                    y_train
                )

                # ---------------------------------------------
                # 4. Evaluate training performance
                # ---------------------------------------------

                train_metrics = self.evaluate_model(
                    best_model,
                    X_train,
                    y_train
                )

                # ---------------------------------------------
                # 5. Save model locally
                # ---------------------------------------------

                self.save_model(
                    best_model
                )

                # ---------------------------------------------
                # 6. Log final training information to MLflow
                # ---------------------------------------------

                mlflow.log_param(
                    "best_model",
                    best_model_name
                )

                mlflow.log_params(
                    best_parameters[
                        best_model_name
                    ]
                )

                mlflow.log_metric(
                    "best_cv_f1",
                    float(
                        cv_scores[
                            best_model_name
                        ]
                    )
                )

                mlflow.log_metrics({
                    "train_accuracy": float(
                        train_metrics["accuracy"]
                    ),

                    "train_precision": float(
                        train_metrics["precision"]
                    ),

                    "train_recall": float(
                        train_metrics["recall"]
                    ),

                    "train_f1": float(
                        train_metrics["f1_score"]
                    ),

                    "train_roc_auc": float(
                        train_metrics["roc_auc"]
                    )
                })

                # ---------------------------------------------
                # 7. Log trained model to MLflow
                # ---------------------------------------------

                mlflow.sklearn.log_model(
                    best_model,
                    "model"
                )

                logging.info(
                    f"Model training completed. "
                    f"Best model: {best_model_name}"
                )

                # ---------------------------------------------
                # 8. Return artifact
                # ---------------------------------------------

                return ModelTrainerArtifact(
                    trained_model_file_path=(
                        self.model_trainer_config
                        .trained_model_file_path
                    ),

                    best_model_name=(
                        best_model_name
                    ),

                    best_model_parameters=(
                        best_parameters[
                            best_model_name
                        ]
                    ),

                    best_cv_score=(
                        cv_scores[
                            best_model_name
                        ]
                    ),

                    train_metric_artifact=(
                        train_metrics
                    )
                )

        except Exception as e:

            raise CustomException(
                e,
                sys
            )