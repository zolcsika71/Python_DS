import numpy as np
import pandas as pd
import re
import logging
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression

from src.config import logger, CONFIG

def try_build_model(prefer_lightgbm: bool):
    """
    Returns a sklearn estimator.
    If LightGBM is available and prefer_lightgbm=True, use it; else LogisticRegression.
    """
    if prefer_lightgbm:
        try:
            from lightgbm import LGBMClassifier
            return LGBMClassifier(**CONFIG.lgbm_params)
        except Exception as e:
            logger.warning(f"LightGBM not usable, falling back to LogisticRegression. Reason: {e}")

    # Strong, simple baseline that works with sparse one-hot features
    return LogisticRegression(**CONFIG.logreg_params)

def clean_column_names_func(df):
    """Function to clean column names for LightGBM compatibility."""
    df = df.copy()
    df.columns = [re.sub(r'[^\w\s]', '', col).replace(' ', '_') for col in df.columns]
    return df

def build_pipeline(cat_cols, num_cols):
    """
    Builds a preprocessing and modeling pipeline.
    """
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=False,
    )
    preprocessor.set_output(transform="pandas")

    return Pipeline([
        ("preprocessor", preprocessor),
        ("clean_names", FunctionTransformer(clean_column_names_func, validate=False))
    ])

def cross_validate_auc(x: pd.DataFrame, y: pd.Series, folds: int, prefer_lightgbm: bool = True):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    arcs = []

    cat_cols = [c for c in x.columns if x[c].dtype == "object"]
    num_cols = [c for c in x.columns if c not in cat_cols]

    for fold, (tr_idx, va_idx) in enumerate(skf.split(x, y), start=1):
        x_tr, x_va = x.iloc[tr_idx], x.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        preprocessor = build_pipeline(cat_cols, num_cols)
        model = try_build_model(prefer_lightgbm=prefer_lightgbm)

        clf = Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", model),
            ]
        )

        clf.fit(x_tr, y_tr)
        proba = clf.predict_proba(x_va)[:, 1]
        auc_val = roc_auc_score(y_va, proba)
        arcs.append(auc_val)
        logger.info(f"CV Fold {fold}/{folds} AUC: {auc_val:.5f}")

    mean_auc = np.mean(arcs)
    std_auc = np.std(arcs)
    logger.info(f"CV Mean AUC: {mean_auc:.5f}  Std: {std_auc:.5f}")
    return float(mean_auc)
