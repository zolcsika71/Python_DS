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

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds engineered features like credit-to-income and annuity-to-income ratios.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Dataframe with new features.
    """
    df = df.copy()
    
    # Credit to Income ratio
    if "AMT_CREDIT" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        df["CREDIT_TO_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
        
    # Annuity to Income ratio
    if "AMT_ANNUITY" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        df["ANNUITY_TO_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
        
    # Goods Price to Credit ratio
    if "AMT_GOODS_PRICE" in df.columns and "AMT_CREDIT" in df.columns:
        df["GOODS_TO_CREDIT_RATIO"] = df["AMT_GOODS_PRICE"] / df["AMT_CREDIT"]
        
    # Days Employed to Days Birth ratio
    if "DAYS_EMPLOYED" in df.columns and "DAYS_BIRTH" in df.columns:
        df["EMPLOYED_TO_BIRTH_RATIO"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
        
    return df

def validate_data_schema(df: pd.DataFrame, expected_columns: list):
    """
    Validates that the dataframe contains expected columns.
    """
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        logger.warning(f"Missing expected columns: {missing_cols}")
    return missing_cols

def check_data_drift(train_df: pd.DataFrame, test_df: pd.DataFrame, threshold: float = 0.1):
    """
    Performs a simple drift check by comparing means of numerical columns.
    """
    num_cols = train_df.select_dtypes(include=[np.number]).columns
    num_cols = [col for col in num_cols if col in test_df.columns and col not in ["SK_ID_CURR", "TARGET"]]
    
    drifted_cols = []
    for col in num_cols:
        train_mean = train_df[col].mean()
        test_mean = test_df[col].mean()
        
        if pd.isna(train_mean) or pd.isna(test_mean) or train_mean == 0:
            continue
            
        diff = abs(train_mean - test_mean) / abs(train_mean)
        if diff > threshold:
            drifted_cols.append((col, diff))
            
    if drifted_cols:
        logger.warning(f"Detected potential data drift in {len(drifted_cols)} columns:")
        for col, diff in drifted_cols[:5]:
            logger.warning(f"  - {col}: relative difference {diff:.2f}")
    else:
        logger.info("No significant data drift detected.")
        
    return drifted_cols
