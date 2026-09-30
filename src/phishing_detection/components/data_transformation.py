import os
import pickle
import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek

from sklearn.impute import SimpleImputer

from phishing_detection.entity.artifact_entity import (
    DataValidationArtifact,
    DataAnalysisArtifact,
    DataTransformationArtifact
)

from phishing_detection.entity.config_entity import (
    DataTransformationConfig
)

from phishing_detection.constants import training_pipeline

from phishing_detection.exception.exception import CustomException
from phishing_detection.logger.logger import logging
import sys

class DataTransformation:

    def __init__(
        self,
        data_validation_artifact: DataValidationArtifact,
        data_analysis_artifact: DataAnalysisArtifact,
        data_transformation_config: DataTransformationConfig
    ):

        self.data_validation_artifact = data_validation_artifact

        self.data_analysis_artifact = data_analysis_artifact

        self.data_transformation_config = (
            data_transformation_config
        )


    def read_data(self):

        train_df = pd.read_csv(
            self.data_validation_artifact.valid_train_file_path
        )

        test_df = pd.read_csv(
            self.data_validation_artifact.valid_test_file_path
        )

        return train_df, test_df

    def separate_features_target(self, train_df, test_df):

        feature_columns = training_pipeline.URL_FEATURE_COLUMNS

        target_column =  training_pipeline.TARGET_COLUMN

        X_train = train_df[feature_columns]
        y_train = train_df[target_column]

        X_test = test_df[feature_columns]
        y_test = test_df[target_column]

        return X_train, y_train, X_test, y_test

    def create_processor(self):

        processor = SimpleImputer(
            strategy="median"
        )

        return processor

    def save_processor(self, processor):

        os.makedirs(
            self.data_transformation_config.data_transformation_dir,
            exist_ok=True
        )

        with open(
            self.data_transformation_config.processor_file_path,
            "wb"
        ) as file:

            pickle.dump(processor, file)

    def save_transformed_data(
        self,
        X_train,
        y_train,
        X_test,
        y_test
    ):

        os.makedirs(
            self.data_transformation_config.transformed_dir,
            exist_ok=True
        )

        train_array = np.c_[
            X_train,
            y_train.to_numpy()
        ]

        test_array = np.c_[
            X_test,
            y_test.to_numpy()
        ]

        np.save(
            self.data_transformation_config.transformed_train_file_path,
            train_array
        )

        np.save(
            self.data_transformation_config.transformed_test_file_path,
            test_array
        )

    def initiate_data_transformation(self):

        try:

            train_df, test_df = self.read_data()

            X_train, y_train, X_test, y_test = (
                self.separate_features_target(
                    train_df,
                    test_df
                )
            )

            processor = self.create_processor()

            processor.fit(X_train)

            X_train_transformed = processor.transform(X_train)

            X_test_transformed = processor.transform(X_test)

            if self.data_analysis_artifact.is_imbalanced:

                smote_tomek = SMOTETomek(
                    random_state=42
                )

                X_train_transformed, y_train = smote_tomek.fit_resample(
                    X_train_transformed,
                    y_train
                )

                logging.info(
                    "Class imbalance detected. SMOTETomek applied."
                )

            else:

                logging.info(
                    "Class imbalance not detected. "
                    "Skipping SMOTETomek."
                )

            self.save_processor(processor)

            self.save_transformed_data(
                X_train_transformed,
                y_train,
                X_test_transformed,
                y_test
            )

            return DataTransformationArtifact(
                transformed_train_file_path=(
                    self.data_transformation_config
                    .transformed_train_file_path
                ),
                transformed_test_file_path=(
                    self.data_transformation_config
                    .transformed_test_file_path
                ),
                processor_file_path=(
                    self.data_transformation_config
                    .processor_file_path
                )
            )

        except Exception as e:

            raise CustomException(e, sys)

    