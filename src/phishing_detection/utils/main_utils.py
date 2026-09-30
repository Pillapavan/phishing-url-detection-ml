import os
import yaml
from pandas.api.types import is_numeric_dtype


def read_yaml_file(file_path: str) -> dict:
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def write_yaml_file(file_path: str, content: dict):

    os.makedirs(
        os.path.dirname(file_path),
        exist_ok=True
    )

    with open(file_path, "w") as file:
        yaml.dump(
            content,
            file,
            default_flow_style=False
        )

def detect_data_drift(train_df, test_df):

    drift_report = {}

    for column in train_df.columns:

        if column not in test_df.columns:
            continue

        # Skip non-numeric columns
        if not is_numeric_dtype(train_df[column]):
            continue

        train_mean = train_df[column].mean()
        test_mean = test_df[column].mean()

        if train_mean == 0:
            drift_score = 0
        else:
            drift_score = abs(
                train_mean - test_mean
            ) / abs(train_mean)

        drift_report[column] = {
            "train_mean": float(train_mean),
            "test_mean": float(test_mean),
            "drift_score": float(drift_score)
        }

    return drift_report