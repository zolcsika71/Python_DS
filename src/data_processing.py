import os
import pandas as pd
import numpy as np
from src.config import logger

def load_data(data_dir: str):
    """
    Loads application_train.csv and application_test.csv from the data directory.
    """
    train_path = os.path.join(data_dir, "application_train.csv")
    test_path = os.path.join(data_dir, "application_test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        logger.error(f"Missing required files in {data_dir}")
        raise FileNotFoundError(
            "Missing required files. Expected:\n"
            f"  {train_path}\n"
            f"  {test_path}\n"
            "Download/unzip Kaggle data into the data/ folder."
        )

    logger.info(f"Loading data from {data_dir}...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    logger.info(f"Loaded train: {train_df.shape}, test: {test_df.shape}")
    return train_df, test_df

def fix_known_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Known issue from common baselines: DAYS_EMPLOYED has placeholder value 365243.
    We convert it to NaN and add a flag feature.
    """
    if "DAYS_EMPLOYED" in df.columns:
        anom_val = 365243
        df = df.copy()
        df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == anom_val).astype(np.int8)
        df.loc[df["DAYS_EMPLOYED"] == anom_val, "DAYS_EMPLOYED"] = np.nan
    return df
