"""
Visualization utilities for the Home Credit Default Risk pipeline.

RISK ASSESSMENT REPORT (2026-01-23):
- Detected potential data drift in 50 columns.
- Key drifted features: FLAG_EMAIL (187%), REG_REGION_NOT_LIVE_REGION (24%), AMT_CREDIT (14%).
- Severity: Moderate. Impact on test set performance expected.
- Recommendations: Monitor feature importance and explore adversarial validation.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import CalibratedClassifierCV
from src.config import logger

def plot_prediction_distribution(probes, out_path, title="Distribution of Predicted Probabilities"):
    """
    Plots the distribution of predicted probabilities.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(probes, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='white')
    plt.title(title)
    plt.xlabel("Probability")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(out_path)
    logger.info(f"Saved prediction distribution plot to {out_path}")
    plt.close()

def plot_roc_curve(y_true, y_probes, out_path):
    """
    Plots the ROC curve.
    """
    fpr, tpr, _ = roc_curve(y_true, y_probes)
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
    logger.info(f"Saved ROC curve plot to {out_path}")
    plt.close()

def plot_feature_importance(clf, top_n=20, out_dir="plots"):
    """
    Extracts feature names from the pipeline and plots feature importance if the model supports it.
    """
    model = clf.named_steps["model"]
    preprocessor = clf.named_steps["prep"]

    # If it's calibrated, we need to extract the base model
    if isinstance(model, CalibratedClassifierCV):
        model = model.calibrated_classifiers_[0].estimator

    # Extract feature names after transformation
    try:
        # Our pipeline structure: prep -> clean_names -> model
        # 'prep' is a ColumnTransformer
        feature_names = preprocessor.get_feature_names_out().tolist()
        
        # Post-process names to match clean_column_names_func
        feature_names = [re.sub(r'[^\w\s]', '', col).replace(' ', '_') for col in feature_names]

    except Exception as e:
        logger.warning(f"Could not extract feature names: {e}. Using generic names.")
        n_features = getattr(model, "n_features_in_", 0)
        if n_features == 0 and hasattr(model, "feature_importances_"):
            n_features = len(model.feature_importances_)
        elif n_features == 0 and hasattr(model, "coef_"):
            n_features = model.coef_.shape[1]
        feature_names = [f"f{i}" for i in range(n_features)]

    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])

    if importances is not None:
        _plot_importance_data(feature_names, importances, top_n, out_dir)
    else:
        logger.warning("Model does not support feature importance/coefficients.")

def _plot_importance_data(feature_names, importances, top_n, out_dir):
    if len(feature_names) != len(importances):
        logger.warning(f"Feature names length ({len(feature_names)}) doesn't match importances length ({len(importances)}). Using generic names.")
        feature_names = [f"f{i}" for i in range(len(importances))]

    feat_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
    feat_imp = feat_imp.sort_values(by="importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, 8))
    plt.barh(feat_imp["feature"], feat_imp["importance"], color='skyblue')
    plt.gca().invert_yaxis()
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Importance Score")
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, "feature_importance.png")
    plt.savefig(plot_path)
    logger.info(f"Saved feature importance plot to {plot_path}")
    plt.close()

def plot_top_10_closest_targets(top_10_df, out_path):
    """
    Visualizes the top 10 TARGET values closest to 1.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plot_df = top_10_df.sort_values(by='TARGET', ascending=False)
    bars = plt.bar(plot_df['SK_ID_CURR'].astype(str), plot_df['TARGET'], color='salmon')
    plt.axhline(y=1, color='r', linestyle='--', label='Target Value 1.0')
    plt.xlabel('SK_ID_CURR (Customer ID)')
    plt.ylabel('TARGET Value (Probability)')
    plt.title('Top 10 TARGET Values Closest to 1')
    plt.xticks(rotation=45)
    plt.ylim(0, 1.1)
    plt.legend()
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(out_path)
    logger.info(f"Top 10 visualization saved to {out_path}")
    plt.close()

def plot_shap_summary(clf, x_sample, out_path):
    """
    Generates a SHAP summary plot.
    """
    try:
        import shap
        
        # Access the model
        model = clf.named_steps["model"]
        if isinstance(model, CalibratedClassifierCV):
            # SHAP works better with the base estimator
            # For CalibratedClassifierCV, we take the first fold's estimator
            base_model = model.calibrated_classifiers_[0].estimator
        else:
            base_model = model
            
        # Transform the sample
        preprocessor = clf.named_steps["prep"]
        cleaner = clf.named_steps["clean_names"]
        x_transformed = cleaner.transform(preprocessor.transform(x_sample))
        
        # Determine the correct explainer
        if hasattr(base_model, "feature_importances_"):
            explainer = shap.TreeExplainer(base_model)
        else:
            explainer = shap.LinearExplainer(base_model, x_transformed)
            
        shap_values = explainer.shap_values(x_transformed)
        
        # shap_values can be a list for multi-class/binary
        if isinstance(shap_values, list):
            # For binary classification, index 1 is usually the positive class
            shap_values_to_plot = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            shap_values_to_plot = shap_values

        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values_to_plot, x_transformed, show=False)
        plt.title("SHAP Feature Importance (Summary Plot)")
        plt.tight_layout()
        plt.savefig(out_path)
        logger.info(f"Saved SHAP summary plot to {out_path}")
        plt.close()
    except Exception as e:
        logger.warning(f"Could not generate SHAP plot: {e}")

def plot_train_test_distribution(train_proba, test_proba, out_path):
    """
    Plots and compares the distribution of predicted probabilities for train and test sets.
    """
    plt.figure(figsize=(12, 7))
    plt.hist(train_proba, bins=50, label="Train", density=True, alpha=0.4, color="blue", histtype='stepfilled')
    plt.hist(test_proba, bins=50, label="Test", density=True, alpha=0.4, color="orange", histtype='stepfilled')
    plt.title("Train vs Test Prediction Distributions")
    plt.xlabel("Predicted Probability (TARGET=1)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    logger.info(f"Saved distribution comparison plot to {out_path}")
    plt.close()
