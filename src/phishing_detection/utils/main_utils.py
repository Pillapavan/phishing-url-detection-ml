import os
import yaml
import numpy as np
from scipy.stats import ks_2samp


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

def detect_data_drift(
    base_df,
    current_df,
    threshold=0.05
) -> dict:

    drift_report = {}

    for column in base_df.columns:

        if column not in current_df.columns:
            drift_report[column] = {
                "drift_detected": True,
                "p_value": 0.0
            }
            continue

        if not np.issubdtype(
            base_df[column].dtype,
            np.number
        ):
            continue

        _, p_value = ks_2samp(
            base_df[column],
            current_df[column]
        )

        drift_report[column] = {
            "drift_detected": bool(p_value < threshold),
            "p_value": float(p_value)
        }

    return drift_report