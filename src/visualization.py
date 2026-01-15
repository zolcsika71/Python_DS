import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
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

    # Extract feature names after transformation
    try:
        if hasattr(preprocessor, "named_steps") and "preprocessor" in preprocessor.named_steps:
            inner_preprocessor = preprocessor.named_steps["preprocessor"]
            feature_names = inner_preprocessor.get_feature_names_out().tolist()
            feature_names = [re.sub(r'[^\w\s]', '', col).replace(' ', '_') for col in feature_names]
        elif hasattr(preprocessor, 'get_feature_names_out'):
            feature_names = preprocessor.get_feature_names_out().tolist()
        else:
            feature_names = []
            transformers = getattr(preprocessor, 'transformers_', [])
            for name, transformer, columns in transformers:
                if name == 'remainder' and transformer == 'drop':
                    continue
                if hasattr(transformer, 'get_feature_names_out'):
                    names = transformer.get_feature_names_out(columns)
                    feature_names.extend(names)
                else:
                    feature_names.extend(columns)
    except Exception as e:
        logger.warning(f"Could not extract feature names: {e}")
        feature_names = [f"f{i}" for i in range(model.n_features_in_)]

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
