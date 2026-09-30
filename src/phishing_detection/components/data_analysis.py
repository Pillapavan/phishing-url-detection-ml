import os
from phishing_detection.utils.main_utils import write_yaml_file
import pandas as pd
from phishing_detection.entity.artifact_entity import DataAnalysisArtifact
from phishing_detection.constants import training_pipeline



class DataAnalysis:

    def __init__(
        self,
        data_validation_artifact,
        data_analysis_config
    ):
        self.data_validation_artifact = data_validation_artifact
        self.data_analysis_config = data_analysis_config

    def read_data(self):
        train_df = pd.read_csv(
            self.data_validation_artifact.valid_train_file_path
        )

        test_df = pd.read_csv(
            self.data_validation_artifact.valid_test_file_path
        )

        return train_df, test_df
    

    def analyze_data(self, train_df):

        missing_percentage = train_df.isnull().mean() * 100

        target_distribution = train_df[training_pipeline.TARGET_COLUMN].value_counts()

        target_percentage = (
            train_df[training_pipeline.TARGET_COLUMN]
            .value_counts(normalize=True)
            .mul(100)
        )

        minority_percentage = target_percentage.min()

        is_imbalanced = (
            minority_percentage
            < self.data_analysis_config.imbalance_threshold * 100
        )

        analysis_report = {
            "missing_percentage": missing_percentage.to_dict(),
            "target_distribution": target_distribution.to_dict(),
            "target_percentage": target_percentage.to_dict(),
            "minority_percentage": float(minority_percentage),
            "is_imbalanced": bool(is_imbalanced)
        }

        return analysis_report, is_imbalanced

    def save_analysis_report(self, analysis_report):

        os.makedirs(
            self.data_analysis_config.data_analysis_dir,
            exist_ok=True
        )

        write_yaml_file(
            self.data_analysis_config.analysis_report_file_path,
            analysis_report
        )

    def initiate_data_analysis(self):

        train_df, test_df = self.read_data()

        analysis_report, is_imbalanced = self.analyze_data(train_df)

        self.save_analysis_report(analysis_report)

        return DataAnalysisArtifact(
            is_imbalanced=is_imbalanced,
            analysis_report_file_path=(
                self.data_analysis_config.analysis_report_file_path
            )
        )