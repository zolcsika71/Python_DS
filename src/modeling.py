import numpy as np
import pandas as pd
import re
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from src.config import logger, CONFIG

def import_lgbm_callback():
    try:
        from lightgbm import early_stopping, log_evaluation
        return [early_stopping(stopping_rounds=100), log_evaluation(period=100)]
    except ImportError:
        return None

def try_build_model(prefer_lightgbm: bool, calibrate: bool = True):
    """
    Returns a sklearn estimator.
    If LightGBM is available and prefer_lightgbm=True, use it; else LogisticRegression.
    """
    if prefer_lightgbm:
        try:
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(**CONFIG.lgbm_params)
        except Exception as e:
            logger.warning(f"LightGBM not usable, falling back to LogisticRegression. Reason: {e}")
            model = LogisticRegression(**CONFIG.logreg_params)
    else:
        model = LogisticRegression(**CONFIG.logreg_params)

    if calibrate:
        # Platt scaling (method='sigmoid') or Isotonic regression (method='isotonic')
        return CalibratedClassifierCV(model, method='sigmoid', cv=3)
    
    return model

def clean_column_names_func(df):
    """Function to clean column names for LightGBM compatibility."""
    df = df.copy()
    df.columns = [re.sub(r'[^\w\s]', '', col).replace(' ', '_') for col in df.columns]
    return df

def build_pipeline(cat_cols, num_cols, prefer_lightgbm: bool = True, calibrate: bool = True):
    """
    Builds a full preprocessing and modeling pipeline.
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

    model = try_build_model(prefer_lightgbm=prefer_lightgbm, calibrate=calibrate)

    return Pipeline([
        ("prep", preprocessor),
        ("clean_names", FunctionTransformer(clean_column_names_func, validate=False)),
        ("model", model)
    ])

def cross_validate_auc(x: pd.DataFrame, y: pd.Series, folds: int, prefer_lightgbm: bool = True, calibrate: bool = True):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    arcs = []

    cat_cols = [c for c in x.columns if x[c].dtype == "object"]
    num_cols = [c for c in x.columns if c not in cat_cols]

    for fold, (tr_idx, va_idx) in enumerate(skf.split(x, y), start=1):
        x_tr, x_va = x.iloc[tr_idx], x.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        clf = build_pipeline(cat_cols, num_cols, prefer_lightgbm=prefer_lightgbm, calibrate=calibrate)

        if prefer_lightgbm:
            # We need to transform the data before passing it to the model for early stopping
            # But the pipeline handles transformation. 
            # To use early stopping in LGBM with sklearn API, we can use fit_params
            
            # Extract the preprocessor from the pipeline
            preprocessor = clf.named_steps["prep"]
            cleaner = clf.named_steps["clean_names"]
            
            # Fit and transform training and validation data
            x_tr_transformed = cleaner.transform(preprocessor.fit_transform(x_tr, y_tr))
            x_va_transformed = cleaner.transform(preprocessor.transform(x_va))
            
            # Access the model (which might be wrapped in CalibratedClassifierCV)
            model_step = clf.named_steps["model"]
            
            if isinstance(model_step, CalibratedClassifierCV):
                # CalibratedClassifierCV doesn't support early_stopping_rounds in fit directly 
                # for its base estimator in a way that's easy to pass through.
                # So we fit the pipeline normally if calibrated.
                clf.fit(x_tr, y_tr)
            else:
                # If not calibrated, we can use early stopping
                callbacks = import_lgbm_callback()
                model_step.fit(
                    x_tr_transformed, y_tr,
                    eval_set=[(x_va_transformed, y_va)],
                    eval_metric="auc",
                    callbacks=callbacks
                )
                # Ensure the pipeline's FunctionTransformer and ColumnTransformer are also "fitted"
                # Actually, preprocessor was already fitted.
                # Just need to make sure clf.predict works correctly.
                # We already fit the model step.
                # To be safe, we can just call clf.fit(x_tr, y_tr) as well, but that's redundant.
                # Since we are using CalibratedClassifierCV by default now, let's keep it simple.
                clf.fit(x_tr, y_tr)
        else:
            clf.fit(x_tr, y_tr)

        proba = clf.predict_proba(x_va)[:, 1]
        auc_val = roc_auc_score(y_va, proba)
        arcs.append(auc_val)
        logger.info(f"CV Fold {fold}/{folds} AUC: {auc_val:.5f}")

    mean_auc = np.mean(arcs)
    std_auc = np.std(arcs)
    logger.info(f"CV Mean AUC: {mean_auc:.5f}  Std: {std_auc:.5f}")
    return float(mean_auc)
