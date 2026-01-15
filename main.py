import argparse
import os
from datetime import datetime

import pandas as pd
from sklearn.pipeline import Pipeline

from src.data_processing import load_data, fix_known_anomalies
from src.modeling import try_build_model, build_pipeline, cross_validate_auc
from src.visualization import (
    plot_feature_importance,
    plot_roc_curve,
    plot_train_test_distribution
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", help="Folder containing Kaggle CSV files")
    parser.add_argument("--folds", type=int, default=3, help="CV folds (3 is a good exam default)")
    parser.add_argument("--prefer-lightgbm", action="store_true", help="Try LightGBM first (recommended)")
    parser.add_argument("--out", default=None, help="Output submission path (CSV)")
    args = parser.parse_args()

    os.makedirs("submissions", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # 1. Load Data
    train_df, test_df = load_data(args.data_dir)

    # 2. Fix Anomalies
    train_df = fix_known_anomalies(train_df)
    test_df = fix_known_anomalies(test_df)

    if "TARGET" not in train_df.columns:
        raise ValueError("Training file must include TARGET column.")

    if "SK_ID_CURR" not in train_df.columns or "SK_ID_CURR" not in test_df.columns:
        raise ValueError("Expected SK_ID_CURR column in both train and test.")

    y = train_df["TARGET"].astype(int)
    test_ids = test_df["SK_ID_CURR"]

    x_train = train_df.drop(columns=["TARGET"])
    x_test = test_df.copy()

    # 3. Cross-validate
    print("[INFO] Running cross-validation...")
    _ = cross_validate_auc(x_train, y, folds=args.folds, prefer_lightgbm=args.prefer_lightgbm)

    # 4. Train Final Model
    print("[INFO] Training final model on full training data...")
    preprocessor = build_pipeline(x_train)
    model = try_build_model(prefer_lightgbm=args.prefer_lightgbm)

    clf = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", model),
        ]
    )

    clf.fit(x_train, y)
    
    # 5. Visualizations & Evaluation
    plot_feature_importance(clf)

    train_proba = clf.predict_proba(x_train)[:, 1]
    test_proba = clf.predict_proba(x_test)[:, 1]

    dist_plot_path = os.path.join("plots", "train_test_distribution_comparison.png")
    plot_train_test_distribution(train_proba, test_proba, dist_plot_path)

    plot_roc_curve(y, train_proba, os.path.join("plots", "train_roc_curve.png"))

    for old_plot in ["train_prediction_distribution.png", "test_prediction_distribution.png"]:
        old_path = os.path.join("plots", old_plot)
        if os.path.exists(old_path):
            os.remove(old_path)

    # 6. Write Submission
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = args.out or os.path.join("submissions", f"submission_{ts}.csv")

    sub = pd.DataFrame({"SK_ID_CURR": test_ids, "TARGET": test_proba})
    sub.to_csv(out_path, index=False)

    print(f"[OK] Wrote submission: {out_path}")
    print(sub.head(5))

if __name__ == "__main__":
    main()
