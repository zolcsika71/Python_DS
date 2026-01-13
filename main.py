import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve, auc
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


def build_pipeline():
    # Identify column types
    # (We'll detect on the fly in main() after reading the dataframe.)
    def make_preprocessor(x: pd.DataFrame):
        cat_cols = [c for c in x.columns if x[c].dtype == "object"]
        num_cols = [c for c in x.columns if c not in cat_cols]

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

    # We build it lazily because we need x columns first.
    return make_preprocessor

def cross_validate_auc(pipeline_builder, x: pd.DataFrame, y: pd.Series, folds: int):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    aucs = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(x, y), start=1):
        x_tr, x_va = x.iloc[tr_idx], x.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        preprocessor = pipeline_builder(x_tr)
        model = try_build_model(prefer_lightgbm=True)

        clf = Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", model),
            ]
        )

        clf.fit(x_tr, y_tr)
        proba = clf.predict_proba(x_va)[:, 1]
        auc_val = roc_auc_score(y_va, proba)
        aucs.append(auc_val)
        print(f"[CV] Fold {fold}/{folds} AUC: {auc_val:.5f}")

    print(f"[CV] Mean AUC: {np.mean(aucs):.5f}  Std: {np.std(aucs):.5f}")
    return float(np.mean(aucs))


def plot_prediction_distribution(probas, out_path):
    """
    Plots the distribution of predicted probabilities.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(probas, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='white')
    plt.title("Distribution of Predicted Probabilities")
    plt.xlabel("Probability")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"[OK] Saved prediction distribution plot to {out_path}")
    plt.close()


def plot_roc_curve(y_true, y_probas, out_path):
    """
    Plots the ROC curve.
    """
    fpr, tpr, _ = roc_curve(y_true, y_probas)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"[OK] Saved ROC curve plot to {out_path}")
    plt.close()


def plot_feature_importance(clf, top_n=20):
    """
    Extracts feature names from the pipeline and plots feature importance if the model supports it.
    """
    model = clf.named_steps["model"]
    preprocessor = clf.named_steps["prep"]

    # Extract feature names after transformation
    try:
        # Get feature names from ColumnTransformer
        feature_names = []
        for name, transformer, columns in preprocessor.transformers_:
            if name == 'remainder' and transformer == 'drop':
                continue
            if hasattr(transformer, 'get_feature_names_out'):
                names = transformer.get_feature_names_out(columns)
                feature_names.extend(names)
            else:
                feature_names.extend(columns)
    except Exception as e:
        print(f"[WARN] Could not extract feature names: {e}")
        feature_names = [f"f{i}" for i in range(model.n_features_in_)]

    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])

    if importances is not None:
        _plot_importance_data(feature_names, importances, top_n)
    else:
        print("[WARN] Model does not support feature importance/coefficients.")


def _plot_importance_data(feature_names, importances, top_n):
    # Match feature names with importances
    if len(feature_names) != len(importances):
        print(f"[WARN] Feature names length ({len(feature_names)}) doesn't match importances length ({len(importances)}). Using generic names.")
        feature_names = [f"f{i}" for i in range(len(importances))]

    feat_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
    feat_imp = feat_imp.sort_values(by="importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, 8))
    plt.barh(feat_imp["feature"], feat_imp["importance"], color='skyblue')
    plt.gca().invert_yaxis()  # Put highest importance at the top
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Importance Score")
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    plot_path = os.path.join("plots", "feature_importance.png")
    plt.savefig(plot_path)
    print(f"[OK] Saved feature importance plot to {plot_path}")
    plt.close()


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

    # Build pipeline builder (needs x columns)
    pipeline_builder = build_pipeline()

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
    
    # Feature importance plot
    plot_feature_importance(clf)

    # Prediction distributions
    train_proba = clf.predict_proba(X)[:, 1]
    test_proba = clf.predict_proba(X_test)[:, 1]

    # Histogram and Density Comparison using Matplotlib
    plt.figure(figsize=(12, 7))
    plt.hist(train_proba, bins=50, label="Train", density=True, alpha=0.4, color="blue", histtype='stepfilled')
    plt.hist(test_proba, bins=50, label="Test", density=True, alpha=0.4, color="orange", histtype='stepfilled')
    plt.title("Train vs Test Prediction Distributions")
    plt.xlabel("Predicted Probability (TARGET=1)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    dist_plot_path = os.path.join("plots", "train_test_distribution_comparison.png")
    plt.savefig(dist_plot_path)
    print(f"[OK] Saved distribution comparison plot to {dist_plot_path}")
    plt.close()

    # ROC curve (using training data as proxy)
    plot_roc_curve(y, train_proba, os.path.join("plots", "train_roc_curve.png"))

    # Remove old/redundant plots if they exist from previous runs
    for old_plot in ["train_prediction_distribution.png", "test_prediction_distribution.png"]:
        old_path = os.path.join("plots", old_plot)
        if os.path.exists(old_path):
            os.remove(old_path)

    # Write submission
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = args.out or os.path.join("submissions", f"submission_{ts}.csv")

    sub = pd.DataFrame({"SK_ID_CURR": test_ids, "TARGET": test_proba})
    sub.to_csv(out_path, index=False)

    print(f"[OK] Wrote submission: {out_path}")
    print(sub.head(5))


if __name__ == "__main__":
    main()
