import os
import sys

import pandas as pd

from phishing_detection.utils.main_utils import (
    read_yaml_file,
    write_yaml_file,
    detect_data_drift
)

from phishing_detection.constants import training_pipeline

from phishing_detection.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
)

from phishing_detection.entity.config_entity import (
    DataValidationConfig
)

from phishing_detection.exception.exception import CustomException
from phishing_detection.logger.logger import logging


class DataValidation:

    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig
    ):

        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_validation_config = data_validation_config

    # ============================================================
    # READ SCHEMA
    # ============================================================

    def read_schema_file(self) -> dict:

        schema = read_yaml_file(
            self.data_validation_config.schema_file_path
        )

        return schema

    # ============================================================
    # READ TRAIN / TEST DATA
    # ============================================================

    def read_data(self):

        train_df = pd.read_csv(
            self.data_ingestion_artifact.trained_file_path
        )

        test_df = pd.read_csv(
            self.data_ingestion_artifact.test_file_path
        )

        return train_df, test_df

    # ============================================================
    # VALIDATE NUMBER OF COLUMNS
    # ============================================================

    def validate_number_of_columns(
        self,
        dataframe: pd.DataFrame,
        expected_number_of_columns: int
    ) -> bool:

        actual_number_of_columns = len(dataframe.columns)

        return actual_number_of_columns == expected_number_of_columns

    # ============================================================
    # VALIDATE COLUMN NAMES
    # ============================================================

    def validate_column_names(
        self,
        dataframe: pd.DataFrame,
        expected_columns: list
    ) -> bool:

        actual_columns = list(dataframe.columns)

        return actual_columns == expected_columns

    # ============================================================
    # VALIDATE DATA TYPES
    # ============================================================

    def validate_column_datatypes(self, dataframe, expected_columns):
        for column_name, expected_dtype in expected_columns.items():

            if column_name not in dataframe.columns:
                logging.error(f"Missing column: {column_name}")
                return False

            actual_dtype = str(dataframe[column_name].dtype)

            # String columns can appear as object, string, or str
            if expected_dtype == "object":
                if actual_dtype not in ["object", "string", "str"]:
                    logging.error(
                        f"Invalid dtype for {column_name}: "
                        f"expected object/string, got {actual_dtype}"
                    )
                    return False

            else:
                if actual_dtype != expected_dtype:
                    logging.error(
                        f"Invalid dtype for {column_name}: "
                        f"expected {expected_dtype}, got {actual_dtype}"
                    )
                    return False

        return True
    
    # ============================================================
    # VALIDATE MISSING VALUES
    # ============================================================

    def validate_missing_values(
        self,
        dataframe: pd.DataFrame
    ) -> bool:

        missing_percentage = dataframe.isnull().mean()

        columns_exceeding_threshold = missing_percentage[
            missing_percentage >
            self.data_validation_config.missing_value_threshold
        ]

        if not columns_exceeding_threshold.empty:

            logging.error(
                "Columns exceeding missing value threshold: "
                f"{columns_exceeding_threshold.to_dict()}"
            )

            return False

        # Target must never contain missing values
        target_column = training_pipeline.TARGET_COLUMN

        if dataframe[target_column].isnull().any():

            logging.error(
                f"Target column {target_column} "
                "contains missing values."
            )

            return False

        return True

    # ============================================================
    # VALIDATE TARGET COLUMN
    # ============================================================

    def validate_target_column(
        self,
        dataframe: pd.DataFrame,
        target_column: str
    ) -> bool:

        if target_column not in dataframe.columns:

            logging.error(
                f"Target column {target_column} not found."
            )

            return False

        if dataframe[target_column].isnull().any():

            logging.error(
                f"Target column {target_column} contains null values."
            )

            return False

        # New dataset uses:
        # 0 = benign
        # 1 = phishing

        unique_values = set(
            dataframe[target_column].dropna().unique()
        )

        if not unique_values.issubset({0, 1}):

            logging.error(
                f"Unexpected target values: {unique_values}"
            )

            return False

        return True

    # ============================================================
    # DATA DRIFT
    # ============================================================

    def detect_drift(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame
    ):

        drift_report = detect_data_drift(
            train_df,
            test_df
        )

        write_yaml_file(
            self.data_validation_config.drift_report_file_path,
            drift_report
        )

        logging.info(
            "Data drift report generated successfully."
        )

        # Drift is reported but does NOT invalidate the dataset.
        return drift_report

    # ============================================================
    # MAIN VALIDATION
    # ============================================================

    def initiate_data_validation(self) -> DataValidationArtifact:

        try:

            logging.info(
                "Starting data validation"
            )

            # ----------------------------------------------------
            # Read data
            # ----------------------------------------------------

            train_df, test_df = self.read_data()

            schema = self.read_schema_file()

            expected_columns = list(
                schema["columns"].keys()
            )

            expected_number_of_columns = len(
                expected_columns
            )

            # ====================================================
            # TRAIN DATA VALIDATION
            # ====================================================

            train_columns_valid = (
                self.validate_number_of_columns(
                    train_df,
                    expected_number_of_columns
                )
            )

            train_names_valid = (
                self.validate_column_names(
                    train_df,
                    expected_columns
                )
            )

            train_dtypes_valid = (
                self.validate_column_datatypes(
                    train_df,
                    schema["columns"]
                )
            )

            train_missing_valid = (
                self.validate_missing_values(
                    train_df
                )
            )

            train_target_valid = (
                self.validate_target_column(
                    train_df,
                    training_pipeline.TARGET_COLUMN
                )
            )

            # ====================================================
            # TEST DATA VALIDATION
            # ====================================================

            test_columns_valid = (
                self.validate_number_of_columns(
                    test_df,
                    expected_number_of_columns
                )
            )

            test_names_valid = (
                self.validate_column_names(
                    test_df,
                    expected_columns
                )
            )

            test_dtypes_valid = (
                self.validate_column_datatypes(
                    test_df,
                    schema["columns"]
                )
            )

            test_missing_valid = (
                self.validate_missing_values(
                    test_df
                )
            )

            test_target_valid = (
                self.validate_target_column(
                    test_df,
                    training_pipeline.TARGET_COLUMN
                )
            )

            # ====================================================
            # DATA DRIFT
            # ====================================================

            self.detect_drift(
                train_df,
                test_df
            )

            # ====================================================
            # FINAL VALIDATION STATUS
            # ====================================================

            validation_status = (
                train_columns_valid
                and train_names_valid
                and train_dtypes_valid
                and train_missing_valid
                and train_target_valid
                and test_columns_valid
                and test_names_valid
                and test_dtypes_valid
                and test_missing_valid
                and test_target_valid
            )

            logging.info(
                f"Data validation status: {validation_status}"
            )

            # ====================================================
            # SAVE VALID / INVALID DATA
            # ====================================================

            if validation_status:

                valid_train_file_path = (
                    self.data_validation_config
                    .valid_train_file_path
                )

                valid_test_file_path = (
                    self.data_validation_config
                    .valid_test_file_path
                )

                os.makedirs(
                    os.path.dirname(
                        valid_train_file_path
                    ),
                    exist_ok=True
                )

                os.makedirs(
                    os.path.dirname(
                        valid_test_file_path
                    ),
                    exist_ok=True
                )

                train_df.to_csv(
                    valid_train_file_path,
                    index=False
                )

                test_df.to_csv(
                    valid_test_file_path,
                    index=False
                )

                invalid_train_file_path = ""
                invalid_test_file_path = ""

                logging.info(
                    "Train and test data passed validation."
                )

            else:

                valid_train_file_path = ""
                valid_test_file_path = ""

                invalid_train_file_path = (
                    self.data_validation_config
                    .invalid_train_file_path
                )

                invalid_test_file_path = (
                    self.data_validation_config
                    .invalid_test_file_path
                )

                os.makedirs(
                    os.path.dirname(
                        invalid_train_file_path
                    ),
                    exist_ok=True
                )

                os.makedirs(
                    os.path.dirname(
                        invalid_test_file_path
                    ),
                    exist_ok=True
                )

                train_df.to_csv(
                    invalid_train_file_path,
                    index=False
                )

                test_df.to_csv(
                    invalid_test_file_path,
                    index=False
                )

                logging.error(
                    "Train and test data failed validation."
                )

            # ====================================================
            # RETURN ARTIFACT
            # ====================================================

            return DataValidationArtifact(

                validation_status=validation_status,

                valid_train_file_path=(
                    valid_train_file_path
                ),

                valid_test_file_path=(
                    valid_test_file_path
                ),

                invalid_train_file_path=(
                    invalid_train_file_path
                ),

                invalid_test_file_path=(
                    invalid_test_file_path
                ),

                drift_report_file_path=(
                    self.data_validation_config
                    .drift_report_file_path
                )
            )

        except Exception as e:

            raise CustomException(e, sys)