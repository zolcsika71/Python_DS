import os
import pandas as pd
import numpy as np
from src.config import logger

def load_data(data_dir: str):
    """
    Loads application_train.csv and application_test.csv from the data directory,
    and joins them with aggregated features from other files.

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
    
    # Process and join supplemental data
    train_df, test_df = join_supplemental_data(train_df, test_df, data_dir)
    
    logger.info(f"Loaded train: {train_df.shape}, test: {test_df.shape}")
    return train_df, test_df

def join_supplemental_data(train: pd.DataFrame, test: pd.DataFrame, data_dir: str):
    """
    Joins aggregated features from bureau, previous_application, etc.
    """
    # 1. Bureau and Bureau Balance
    bureau_path = os.path.join(data_dir, "bureau.csv")
    bureau_bal_path = os.path.join(data_dir, "bureau_balance.csv")
    
    if os.path.exists(bureau_path):
        logger.info("Processing bureau.csv...")
        bureau = pd.read_csv(bureau_path)
        
        if os.path.exists(bureau_bal_path):
            logger.info("Processing bureau_balance.csv...")
            bb = pd.read_csv(bureau_bal_path)
            bb_agg = bb.groupby("SK_ID_BUREAU").agg({
                "MONTHS_BALANCE": ["min", "max", "size"]
            })
            bb_agg.columns = ["BB_" + "_".join(x).upper() for x in bb_agg.columns.ravel()]
            bb_agg.reset_index(inplace=True)
            bureau = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")
            logger.info("Bureau balance features merged into bureau.")

        # Simple aggregations
        bureau_agg = bureau.groupby("SK_ID_CURR").agg({
            "SK_ID_BUREAU": "count",
            "DAYS_CREDIT": ["min", "max", "mean"],
            "CREDIT_DAY_OVERDUE": ["max", "mean"],
            "AMT_CREDIT_SUM": ["max", "mean", "sum"],
        })
        bureau_agg.columns = ["_".join(x).upper() for x in bureau_agg.columns.ravel()]
        bureau_agg.reset_index(inplace=True)
        
        train = train.merge(bureau_agg, on="SK_ID_CURR", how="left")
        test = test.merge(bureau_agg, on="SK_ID_CURR", how="left")
        logger.info(f"Bureau features added. New shape: {train.shape}")

    # 2. Previous Applications
    prev_path = os.path.join(data_dir, "previous_application.csv")
    if os.path.exists(prev_path):
        logger.info("Processing previous_application.csv...")
        prev = pd.read_csv(prev_path)
        
        # Simple aggregations
        prev_agg = prev.groupby("SK_ID_CURR").agg({
            "SK_ID_PREV": "count",
            "AMT_ANNUITY": ["max", "mean"],
            "AMT_APPLICATION": ["max", "mean", "sum"],
            "AMT_CREDIT": ["max", "mean", "sum"],
            "DAYS_DECISION": ["min", "max", "mean"],
            "CNT_PAYMENT": ["mean", "sum"],
        })
        prev_agg.columns = ["PREV_" + "_".join(x).upper() for x in prev_agg.columns.ravel()]
        prev_agg.reset_index(inplace=True)
        
        train = train.merge(prev_agg, on="SK_ID_CURR", how="left")
        test = test.merge(prev_agg, on="SK_ID_CURR", how="left")
        logger.info(f"Previous application features added. New shape: {train.shape}")

    # 3. POS CASH Balance
    pos_path = os.path.join(data_dir, "POS_CASH_balance.csv")
    if os.path.exists(pos_path):
        logger.info("Processing POS_CASH_balance.csv...")
        pos = pd.read_csv(pos_path)
        pos_agg = pos.groupby("SK_ID_CURR").agg({
            "MONTHS_BALANCE": ["max", "mean", "size"],
            "SK_DPD": ["max", "mean"],
            "SK_DPD_DEF": ["max", "mean"],
        })
        pos_agg.columns = ["POS_" + "_".join(x).upper() for x in pos_agg.columns.ravel()]
        pos_agg.reset_index(inplace=True)
        
        train = train.merge(pos_agg, on="SK_ID_CURR", how="left")
        test = test.merge(pos_agg, on="SK_ID_CURR", how="left")
        logger.info(f"POS features added. New shape: {train.shape}")

    # 4. Installments Payments
    inst_path = os.path.join(data_dir, "installments_payments.csv")
    if os.path.exists(inst_path):
        logger.info("Processing installments_payments.csv...")
        inst = pd.read_csv(inst_path)
        # Calculate delay
        inst["PAYMENT_DELAY"] = inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]
        inst["PAYMENT_DIFF"] = inst["AMT_INSTALMENT"] - inst["AMT_PAYMENT"]
        
        inst_agg = inst.groupby("SK_ID_CURR").agg({
            "NUM_INSTALMENT_VERSION": ["nunique"],
            "PAYMENT_DELAY": ["max", "mean", "sum"],
            "PAYMENT_DIFF": ["max", "mean", "sum"],
            "AMT_INSTALMENT": ["max", "mean", "sum"],
            "AMT_PAYMENT": ["min", "max", "mean", "sum"],
            "DAYS_ENTRY_PAYMENT": ["max", "mean", "sum"],
        })
        inst_agg.columns = ["INSTAL_" + "_".join(x).upper() for x in inst_agg.columns.ravel()]
        inst_agg.reset_index(inplace=True)
        
        train = train.merge(inst_agg, on="SK_ID_CURR", how="left")
        test = test.merge(inst_agg, on="SK_ID_CURR", how="left")
        logger.info(f"Installments features added. New shape: {train.shape}")

    # 5. Credit Card Balance
    cc_path = os.path.join(data_dir, "credit_card_balance.csv")
    if os.path.exists(cc_path):
        logger.info("Processing credit_card_balance.csv...")
        cc = pd.read_csv(cc_path)
        cc_agg = cc.groupby("SK_ID_CURR").agg({
            "AMT_BALANCE": ["max", "mean", "sum"],
            "AMT_CREDIT_LIMIT_ACTUAL": ["max", "mean"],
            "AMT_DRAWINGS_ATM_CURRENT": ["max", "mean", "sum"],
            "AMT_DRAWINGS_CURRENT": ["max", "mean", "sum"],
            "SK_DPD": ["max", "mean"],
            "SK_DPD_DEF": ["max", "mean"],
        })
        cc_agg.columns = ["CC_" + "_".join(x).upper() for x in cc_agg.columns.ravel()]
        cc_agg.reset_index(inplace=True)
        
        train = train.merge(cc_agg, on="SK_ID_CURR", how="left")
        test = test.merge(cc_agg, on="SK_ID_CURR", how="left")
        logger.info(f"Credit card features added. New shape: {train.shape}")

    return train, test

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
