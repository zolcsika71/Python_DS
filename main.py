import argparse
from src.orchestrator import run_pipeline

def main():
    parser = argparse.ArgumentParser(description="Home Credit Default Risk Pipeline")
    parser.add_argument("--data-dir", default=None, help="Folder containing Kaggle CSV files")
    parser.add_argument("--folds", type=int, default=3, help="CV folds")
    parser.add_argument("--prefer-lightgbm", action="store_true", help="Try LightGBM first")
    parser.add_argument("--out", default=None, help="Output submission path (CSV)")
    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        folds=args.folds,
        prefer_lightgbm=args.prefer_lightgbm,
        custom_out=args.out
    )

if __name__ == "__main__":
    main()
