import os
import pandas as pd
import numpy as np
from src.config import logger

def load_data(data_dir: str):
    """
    Loads application_train.csv and application_test.csv from the data directory.

    Args:
        data_dir (str): Path to the directory containing the CSV files.

    Returns:
        tuple: (train_df, test_df) as pandas DataFrames.
    """
    files = {
        "train": os.path.join(data_dir, "application_train.csv"),
        "test": os.path.join(data_dir, "application_test.csv")
    }

    for name, path in files.items():
        if not os.path.exists(path):
            logger.error(f"Missing {name} file: {path}")
            raise FileNotFoundError(
                f"Missing required {name} file at {path}. "
                "Ensure Kaggle data is in the data/ folder."
            )

    logger.info(f"Loading data from {data_dir}...")
    train_df = pd.read_csv(files["train"])
    test_df = pd.read_csv(files["test"])
    logger.info(f"Loaded train: {train_df.shape}, test: {test_df.shape}")
    return train_df, test_df

def fix_known_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Known issue from common baselines: DAYS_EMPLOYED has placeholder value 365243.
    We convert it to NaN and add a flag feature.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Dataframe with anomalies fixed.
    """
    if "DAYS_EMPLOYED" in df.columns:
        anom_val = 365243
        df = df.copy()
        df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == anom_val).astype(np.int8)
        df.loc[df["DAYS_EMPLOYED"] == anom_val, "DAYS_EMPLOYED"] = np.nan
    return df
