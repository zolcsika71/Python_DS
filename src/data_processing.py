import os
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from src.config import logger

def load_data(data_dir: str):
    """
    Loads application_train.csv and application_test.csv from the data directory,
    and joins them with aggregated features from other files.
    Optimized to handle large CSV files by leveraging efficient merging and parallel processing.

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
    # Load main application files
    train_df = pd.read_csv(files["train"])
    test_df = pd.read_csv(files["test"])
    
    # Process and join supplemental data (bureau, previous apps, etc.) in parallel
    train_df, test_df = join_supplemental_data(train_df, test_df, data_dir)
    
    logger.info(f"Loaded train: {train_df.shape}, test: {test_df.shape}")
    return train_df, test_df

def _read_csv(path):
    """Memory-efficient CSV reader with specified dtypes."""
    # Optimization: Read a small sample to infer types, then optimize them
    df_sample = pd.read_csv(path, nrows=100)
    dtypes = {}
    for col in df_sample.columns:
        if df_sample[col].dtype == "float64":
            dtypes[col] = "float32"
        elif df_sample[col].dtype == "int64":
            if col.startswith("SK_ID"):
                dtypes[col] = "int32"
            else:
                dtypes[col] = "int32"
    
    return pd.read_csv(path, dtype=dtypes)

def _process_bureau(data_dir):
    bureau_path = os.path.join(data_dir, "bureau.csv")
    bureau_bal_path = os.path.join(data_dir, "bureau_balance.csv")
    if not os.path.exists(bureau_path):
        return None
    
    logger.info("Processing bureau.csv...")
    bureau = _read_csv(bureau_path)
    
    if os.path.exists(bureau_bal_path):
        logger.info("Processing bureau_balance.csv...")
        bb = _read_csv(bureau_bal_path)
        bb_agg = bb.groupby("SK_ID_BUREAU").agg({
            "MONTHS_BALANCE": ["min", "max", "size"]
        })
        bb_agg.columns = ["BB_" + "_".join(x).upper() for x in bb_agg.columns]
        bureau = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")

    bureau_agg = bureau.groupby("SK_ID_CURR").agg({
        "SK_ID_BUREAU": "count",
        "DAYS_CREDIT": ["min", "max", "mean"],
        "CREDIT_DAY_OVERDUE": ["max", "mean"],
        "AMT_CREDIT_SUM": ["max", "mean", "sum"],
    })
    bureau_agg.columns = ["_".join(x).upper() for x in bureau_agg.columns]
    return bureau_agg

def _process_previous_applications(data_dir):
    prev_path = os.path.join(data_dir, "previous_application.csv")
    if not os.path.exists(prev_path):
        return None
    
    logger.info("Processing previous_application.csv...")
    prev = _read_csv(prev_path)
    prev_agg = prev.groupby("SK_ID_CURR").agg({
        "SK_ID_PREV": "count",
        "AMT_ANNUITY": ["max", "mean"],
        "AMT_APPLICATION": ["max", "mean", "sum"],
        "AMT_CREDIT": ["max", "mean", "sum"],
        "DAYS_DECISION": ["min", "max", "mean"],
        "CNT_PAYMENT": ["mean", "sum"],
    })
    prev_agg.columns = ["PREV_" + "_".join(x).upper() for x in prev_agg.columns]
    return prev_agg

def _process_pos_cash(data_dir):
    pos_path = os.path.join(data_dir, "POS_CASH_balance.csv")
    if not os.path.exists(pos_path):
        return None
    
    logger.info("Processing POS_CASH_balance.csv...")
    pos = _read_csv(pos_path)
    pos_agg = pos.groupby("SK_ID_CURR").agg({
        "MONTHS_BALANCE": ["max", "mean", "size"],
        "SK_DPD": ["max", "mean"],
        "SK_DPD_DEF": ["max", "mean"],
    })
    pos_agg.columns = ["POS_" + "_".join(x).upper() for x in pos_agg.columns]
    return pos_agg

def _process_installments(data_dir):
    inst_path = os.path.join(data_dir, "installments_payments.csv")
    if not os.path.exists(inst_path):
        return None
    
    logger.info("Processing installments_payments.csv...")
    inst = _read_csv(inst_path)
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
    inst_agg.columns = ["INSTAL_" + "_".join(x).upper() for x in inst_agg.columns]
    return inst_agg

def _process_credit_card(data_dir):
    cc_path = os.path.join(data_dir, "credit_card_balance.csv")
    if not os.path.exists(cc_path):
        return None
    
    logger.info("Processing credit_card_balance.csv...")
    cc = _read_csv(cc_path)
    cc_agg = cc.groupby("SK_ID_CURR").agg({
        "AMT_BALANCE": ["max", "mean", "sum"],
        "AMT_CREDIT_LIMIT_ACTUAL": ["max", "mean"],
        "AMT_DRAWINGS_ATM_CURRENT": ["max", "mean", "sum"],
        "AMT_DRAWINGS_CURRENT": ["max", "mean", "sum"],
        "SK_DPD": ["max", "mean"],
        "SK_DPD_DEF": ["max", "mean"],
    })
    cc_agg.columns = ["CC_" + "_".join(x).upper() for x in cc_agg.columns]
    return cc_agg

def join_supplemental_data(train: pd.DataFrame, test: pd.DataFrame, data_dir: str):
    """
    Joins aggregated features from bureau, previous_application, etc. in parallel.
    This function performs a series of relational joins to enrich the main application
    tables with historical and behavioral data.
    """
    process_funcs = [
        _process_bureau,
        _process_previous_applications,
        _process_pos_cash,
        _process_installments,
        _process_credit_card
    ]
    
    logger.info(f"Starting parallel processing of supplemental data with {len(process_funcs)} workers...")
    
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(func, data_dir) for func in process_funcs]
        results = [f.result() for f in futures]
    
    for agg_df in results:
        if agg_df is not None:
            train = train.merge(agg_df, on="SK_ID_CURR", how="left")
            test = test.merge(agg_df, on="SK_ID_CURR", how="left")
            
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
    Uses in-place assignment to avoid redundant copies for large dataframes.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Dataframe with new features.
    """
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
        # Sort by drift score descending for clearer logs
        drifted_cols_sorted = sorted(drifted_cols, key=lambda x: x[1], reverse=True)
        for col, diff in drifted_cols_sorted[:5]:
            logger.warning(f"  - {col}: relative difference {diff:.2f}")
    else:
        logger.info("No significant data drift detected.")
        
    return drifted_cols

def select_features_by_drift(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                            drift_threshold: float = 0.1, importance_threshold: float = 10.0,
                            importances: pd.DataFrame = None):
    """
    Identifies and drops features that show significant drift and have low importance.
    
    Args:
        train_df (pd.DataFrame): Training data.
        test_df (pd.DataFrame): Test data.
        drift_threshold (float): Threshold for relative mean difference.
        importance_threshold (float): Threshold for feature importance.
        importances (pd.DataFrame): DataFrame with 'feature' and 'importance' columns.
        
    Returns:
        tuple: (train_df, test_df, dropped_features)
    """
    drifted_results = check_data_drift(train_df, test_df, threshold=drift_threshold)
    drifted_cols = dict(drifted_results)

    to_drop = []

    if importances is not None:
        # Map back to original features (some might be OHE)
        # But here check_data_drift works on original columns before OHE.
        for col, drift_score in drifted_cols.items():
            # Find importance for this feature
            feat_importance = importances[importances['feature'] == col]['importance'].max()

            # If importance is NaN (not in model) or below threshold, drop it
            if pd.isna(feat_importance) or feat_importance < importance_threshold:
                to_drop.append(col)
                logger.info(f"Dropping drifted feature: {col} (Drift: {drift_score:.2f}, Importance: {feat_importance})")
    else:
        # If no importances provided, we can't safely drop based on importance
        # Maybe just log?
        logger.warning("No importance data provided for drift-based feature selection.")

    if to_drop:
        train_df = train_df.drop(columns=to_drop)
        test_df = test_df.drop(columns=to_drop)
        logger.info(f"Dropped {len(to_drop)} features due to drift and low importance.")

    return train_df, test_df, to_drop
