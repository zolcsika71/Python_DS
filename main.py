import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression


def try_build_model(prefer_lightgbm: bool):
    """
    Returns a sklearn estimator.
    If LightGBM is available and prefer_lightgbm=True, use it; else LogisticRegression.
    """
    if prefer_lightgbm:
        try:
            from lightgbm import LGBMClassifier
            return LGBMClassifier(
                n_estimators=800,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                n_jobs=-1,
                random_state=42,
            )
        except Exception as e:
            print(f"[WARN] LightGBM not usable, falling back to LogisticRegression. Reason: {e}")

    # Strong, simple baseline that works with sparse one-hot features
    return LogisticRegression(
        solver="saga",
        max_iter=400,
        n_jobs=-1,
        # class_weight="balanced",  # optional: try if you want
        random_state=42,
    )


def load_data(data_dir: str):
    train_path = os.path.join(data_dir, "application_train.csv")
    test_path = os.path.join(data_dir, "application_test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            "Missing required files. Expected:\n"
            f"  {train_path}\n"
            f"  {test_path}\n"
            "Download/unzip Kaggle data into the data/ folder."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def fix_known_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Known issue from common baselines: DAYS_EMPLOYED has a placeholder value 365243.
    We convert it to NaN and add a flag feature.
    """
    if "DAYS_EMPLOYED" in df.columns:
        anom_val = 365243
        df = df.copy()
        df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == anom_val).astype(np.int8)
        df.loc[df["DAYS_EMPLOYED"] == anom_val, "DAYS_EMPLOYED"] = np.nan
    return df


def build_pipeline(model):
    # Identify column types
    # (We'll detect on the fly in main() after reading the dataframe.)
    def make_preprocessor(X: pd.DataFrame):
        cat_cols = [c for c in X.columns if X[c].dtype == "object"]
        num_cols = [c for c in X.columns if c not in cat_cols]

        numeric = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )

        categorical = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric, num_cols),
                ("cat", categorical, cat_cols),
            ],
            remainder="drop",
            sparse_threshold=0.3,
        )
        return preprocessor

    # We build it lazily because we need X columns first.
    return make_preprocessor

def cross_validate_auc(pipeline_builder, X: pd.DataFrame, y: pd.Series, folds: int):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    aucs = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        preprocessor = pipeline_builder(X_tr)
        model = try_build_model(prefer_lightgbm=True)

        clf = Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", model),
            ]
        )

        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_va)[:, 1]
        auc = roc_auc_score(y_va, proba)
        aucs.append(auc)
        print(f"[CV] Fold {fold}/{folds} AUC: {auc:.5f}")

    print(f"[CV] Mean AUC: {np.mean(aucs):.5f}  Std: {np.std(aucs):.5f}")
    return float(np.mean(aucs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", help="Folder containing Kaggle CSV files")
    parser.add_argument("--folds", type=int, default=3, help="CV folds (3 is a good exam default)")
    parser.add_argument("--prefer-lightgbm", action="store_true", help="Try LightGBM first (recommended)")
    parser.add_argument("--out", default=None, help="Output submission path (CSV)")
    args = parser.parse_args()

    os.makedirs("submissions", exist_ok=True)

    train_df, test_df = load_data(args.data_dir)

    train_df = fix_known_anomalies(train_df)
    test_df = fix_known_anomalies(test_df)

    if "TARGET" not in train_df.columns:
        raise ValueError("Training file must include TARGET column.")

    if "SK_ID_CURR" not in train_df.columns or "SK_ID_CURR" not in test_df.columns:
        raise ValueError("Expected SK_ID_CURR column in both train and test.")

    y = train_df["TARGET"].astype(int)
    train_ids = train_df["SK_ID_CURR"]
    test_ids = test_df["SK_ID_CURR"]

    X = train_df.drop(columns=["TARGET"])
    X_test = test_df.copy()

    # Build pipeline builder (needs X columns)
    pipeline_builder = build_pipeline(model=None)

    # Cross-validate (optional but very useful in an exam to show methodology)
    print("[INFO] Running cross-validation...")
    _ = cross_validate_auc(pipeline_builder, X, y, folds=args.folds)

    # Train final model on full data
    print("[INFO] Training final model on full training data...")
    preprocessor = pipeline_builder(X)
    model = try_build_model(prefer_lightgbm=args.prefer_lightgbm)

    clf = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", model),
        ]
    )

    clf.fit(X, y)
    test_proba = clf.predict_proba(X_test)[:, 1]

    # Write submission
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = args.out or os.path.join("submissions", f"submission_{ts}.csv")

    sub = pd.DataFrame({"SK_ID_CURR": test_ids, "TARGET": test_proba})
    sub.to_csv(out_path, index=False)

    print(f"[OK] Wrote submission: {out_path}")
    print(sub.head(5))


if __name__ == "__main__":
    main()

