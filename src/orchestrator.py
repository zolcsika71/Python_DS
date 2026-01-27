import os
from datetime import datetime

import numpy as np
import pandas as pd

from src.config import logger, CONFIG, ModelConfig, setup_directories
from src.data_processing import (
    load_data,
    fix_known_anomalies,
    add_engineered_features,
    validate_data_schema,
    select_features_by_drift
)
from src.modeling import build_pipeline, cross_validate_auc
from src.visualization import (
    plot_feature_importance,
    plot_roc_curve,
    plot_train_test_distribution,
    plot_top_10_closest_targets,
    plot_shap_summary
)


def analyze_top_10_targets(submission_df: pd.DataFrame, config: ModelConfig):
    """
    Identifies and visualizes the top 10 TARGET values closest to 1.

    Args:
        submission_df (pd.DataFrame): The model predictions.
        config (ModelConfig): Project configuration.

    Returns:
        pd.DataFrame: The top 10 closest targets.
    """
    logger.info("Identifying top 10 TARGET values closest to 1...")
    df = submission_df.copy()
    df['dist_to_1'] = (df['TARGET'] - 1).abs()
    
    top_10_df = df.nsmallest(10, 'dist_to_1').copy()
    top_10_cleaned = top_10_df[['SK_ID_CURR', 'TARGET']].reset_index(drop=True)
    
    # Visualization
    plot_path = os.path.join(config.paths.plots_dir, "top_10_targets_closest_to_1.png")
    plot_top_10_closest_targets(top_10_cleaned, plot_path)

    output_csv = os.path.join(config.paths.submissions_dir, "top_10_closest_targets.csv")
    top_10_cleaned.to_csv(output_csv, index=False)
    logger.info(f"Top 10 dataset saved to {output_csv}")
    
    return top_10_cleaned

def run_pipeline(data_dir=None, folds=3, prefer_lightgbm=True, custom_out=None, use_ensemble=True, config: ModelConfig = CONFIG):
    """
    Orchestrates the full machine learning pipeline.
    
    Workflow Sequence:
    1. Data Loading: Parallel ingestion of relational datasets.
    2. Feature Engineering: Domain-specific ratios and anomaly correction.
    3. Informed Drift Mitigation: Pilot-model pass to filter unstable features.
    4. Model Training: Ensemble stacking with stratified cross-validation.
    5. Evaluation: ROC, SHAP, and probability distribution analysis.
    
    Args:
        data_dir (str, optional): Custom path to data directory.
        folds (int): Number of cross-validation folds.
        prefer_lightgbm (bool): Whether to try LightGBM first.
        custom_out (str, optional): Custom path for the output submission file.
        use_ensemble (bool): Whether to use the ensemble stack (default True for Competition Grade).
        config (ModelConfig): Project configuration.
    """
    setup_directories(config)
    data_dir = data_dir or config.paths.data_dir

    # 1. Load Data
    train_df, test_df = load_data(data_dir)

    # 2. Validation & Preprocessing
    validate_data_schema(train_df, [config.target_col, config.id_col])
    
    # Informed Feature Selection Logic:
    # Instead of blindly dropping drifted features (which could be highly predictive), 
    # we run a 'pilot' model to weigh their importance against their instability.
    logger.info("Performing informed feature selection based on data drift...")
    
    # Standardize data state before selection
    train_df = fix_known_anomalies(train_df)
    train_df = add_engineered_features(train_df)
    test_df = fix_known_anomalies(test_df)
    test_df = add_engineered_features(test_df)
    
    y_temp = train_df[config.target_col].astype(np.int8)
    x_temp = train_df.drop(columns=[config.target_col])
    
    # Optimization: Use a sample for the pilot run to save compute time
    sample_size = min(50000, len(x_temp))
    x_sample = x_temp.iloc[:sample_size]
    y_sample = y_temp.iloc[:sample_size]
    
    cat_cols_temp = [c for c in x_sample.columns if x_sample[c].dtype == "object"]
    num_cols_temp = [c for c in x_sample.columns if c not in cat_cols_temp]
    
    # Build a fast, non-calibrated pipeline for importance estimation
    temp_clf = build_pipeline(cat_cols_temp, num_cols_temp, prefer_lightgbm=prefer_lightgbm, calibrate=False)
    temp_clf.fit(x_sample, y_sample)
    
    # Extract feature importances from the pilot model
    model = temp_clf.named_steps["model"]
    preprocessor = temp_clf.named_steps["prep"]
    feature_names = preprocessor.get_feature_names_out().tolist()
    importances_vals = model.feature_importances_ if hasattr(model, "feature_importances_") else np.abs(model.coef_[0])
    
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances_vals})
    
    # Apply the drift-importance filter
    train_df, test_df, dropped = select_features_by_drift(train_df, test_df, importances=importance_df)
    
    if dropped:
        logger.info(f"Fixed data drift issues by dropping {len(dropped)} problematic features.")

    target_col = config.target_col
    id_col = config.id_col

    if target_col not in train_df.columns:
        raise ValueError(f"Training file must include {target_col} column.")

    if id_col not in train_df.columns or id_col not in test_df.columns:
        raise ValueError(f"Expected {id_col} column in both train and test.")

    y = train_df[target_col].astype(np.int8)
    test_ids = test_df[id_col]

    x_train = train_df.drop(columns=[target_col])
    x_test = test_df.copy()

    # 3. Cross-validate
    logger.info(f"Running {folds}-fold cross-validation...")
    _ = cross_validate_auc(x_train, y, folds=folds, prefer_lightgbm=prefer_lightgbm, use_ensemble=use_ensemble)

    # 4. Train Final Model
    logger.info("Training final model on full training data...")
    cat_cols = [c for c in x_train.columns if x_train[c].dtype == "object"]
    num_cols = [c for c in x_train.columns if c not in cat_cols]
    
    clf = build_pipeline(cat_cols, num_cols, prefer_lightgbm=prefer_lightgbm, use_ensemble=use_ensemble)
    clf.fit(x_train, y)
    
    # 5. Visualizations & Evaluation
    plots_dir = config.paths.plots_dir
    plot_feature_importance(clf, out_dir=plots_dir)
    
    # SHAP explainability (using a sample to speed up)
    shap_sample = x_train.sample(min(100, len(x_train)), random_state=42)
    shap_plot_path = os.path.join(plots_dir, "shap_summary.png")
    plot_shap_summary(clf, shap_sample, shap_plot_path)

    train_proba = clf.predict_proba(x_train)[:, 1]
    test_proba = clf.predict_proba(x_test)[:, 1]

    dist_plot_path = os.path.join(plots_dir, "train_test_distribution_comparison.png")
    plot_train_test_distribution(train_proba, test_proba, dist_plot_path)

    roc_plot_path = os.path.join(plots_dir, "train_roc_curve.png")
    plot_roc_curve(y, train_proba, roc_plot_path)

    # 6. Write Submission
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = custom_out or os.path.join(config.paths.submissions_dir, f"submission_{ts}.csv")

    sub = pd.DataFrame({id_col: test_ids, target_col: test_proba})
    sub.to_csv(out_path, index=False)

    logger.info(f"Wrote submission: {out_path}")
    
    # 7. Analyze Top 10 Closest Targets (CLI output requirement)
    top_10 = analyze_top_10_targets(sub, config)
    print("\nTop 10 TARGET values closest to 1 (from top_10_closest_targets.csv):")
    print(top_10.to_string(index=False))

    return out_path
