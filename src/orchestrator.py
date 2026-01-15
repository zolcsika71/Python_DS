import os
from datetime import datetime
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import logger, TARGET_COL, ID_COL, PLOTS_DIR, SUBMISSIONS_DIR, setup_directories
from src.data_processing import load_data, fix_known_anomalies
from src.modeling import try_build_model, build_pipeline, cross_validate_auc
from src.visualization import (
    plot_feature_importance,
    plot_roc_curve,
    plot_train_test_distribution
)

def run_pipeline(data_dir, folds, prefer_lightgbm, custom_out=None):
    """
    Orchestrates the full machine learning pipeline.
    """
    setup_directories()

    # 1. Load Data
    train_df, test_df = load_data(data_dir)

    # 2. Fix Anomalies
    train_df = fix_known_anomalies(train_df)
    test_df = fix_known_anomalies(test_df)

    if TARGET_COL not in train_df.columns:
        raise ValueError(f"Training file must include {TARGET_COL} column.")

    if ID_COL not in train_df.columns or ID_COL not in test_df.columns:
        raise ValueError(f"Expected {ID_COL} column in both train and test.")

    y = train_df[TARGET_COL].astype(int)
    test_ids = test_df[ID_COL]

    x_train = train_df.drop(columns=[TARGET_COL])
    x_test = test_df.copy()

    # 3. Cross-validate
    logger.info("Running cross-validation...")
    _ = cross_validate_auc(x_train, y, folds=folds, prefer_lightgbm=prefer_lightgbm)

    # 4. Train Final Model
    logger.info("Training final model on full training data...")
    preprocessor = build_pipeline(x_train)
    model = try_build_model(prefer_lightgbm=prefer_lightgbm)

    clf = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", model),
        ]
    )

    clf.fit(x_train, y)
    
    # 5. Visualizations & Evaluation
    plot_feature_importance(clf, out_dir=PLOTS_DIR)

    train_proba = clf.predict_proba(x_train)[:, 1]
    test_proba = clf.predict_proba(x_test)[:, 1]

    dist_plot_path = os.path.join(PLOTS_DIR, "train_test_distribution_comparison.png")
    plot_train_test_distribution(train_proba, test_proba, dist_plot_path)

    roc_plot_path = os.path.join(PLOTS_DIR, "train_roc_curve.png")
    plot_roc_curve(y, train_proba, roc_plot_path)

    # Cleanup old plots if they exist
    for old_plot in ["train_prediction_distribution.png", "test_prediction_distribution.png"]:
        old_path = os.path.join(PLOTS_DIR, old_plot)
        if os.path.exists(old_path):
            os.remove(old_path)

    # 6. Write Submission
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = custom_out or os.path.join(SUBMISSIONS_DIR, f"submission_{ts}.csv")

    sub = pd.DataFrame({ID_COL: test_ids, TARGET_COL: test_proba})
    sub.to_csv(out_path, index=False)

    logger.info(f"Wrote submission: {out_path}")
    print(sub.head(5))

    return out_path
