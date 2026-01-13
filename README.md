# Python Data Science Project - Home Credit Default Risk

This project provides a pipeline for the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) Kaggle competition. It includes data loading, preprocessing (handling anomalies, imputation, and encoding), model training (Logistic Regression or LightGBM), cross-validation, and visualization of results.

## Requirements

- Python >= 3.12
- [Poetry](https://python-poetry.org/) for dependency management.

### Dataset

The project expects Kaggle's "Home Credit Default Risk" dataset.
- Place `application_train.csv` and `application_test.csv` in the `data/` directory.
- You can override the data directory using the `--data-dir` flag.

## Setup

1.  **Install Dependencies**:
    ```bash
    poetry install
    ```

## Usage

### Running the Main Script

Execute the training and prediction pipeline:

```bash
poetry run python main.py [OPTIONS]
```

### Command Line Arguments

- `--data-dir`: Folder containing Kaggle CSV files (default: `data`).
- `--folds`: Number of Cross-Validation folds (default: `3`).
- `--prefer-lightgbm`: Try to use LightGBM if available (recommended).
- `--out`: Path to save the output submission CSV. If not provided, it generates a timestamped file in `submissions/`.

### Example Commands

- **Basic run with Logistic Regression**:
  ```bash
  poetry run python main.py
  ```

- **Run with LightGBM and 5-fold CV**:
  ```bash
  poetry run python main.py --prefer-lightgbm --folds 5
  ```

## Features

- **Data Preprocessing**: Automatically handles categorical and numerical columns.
- **Anomaly Handling**: Fixes known anomalies like `DAYS_EMPLOYED = 365243`.
- **Reproducibility**: Uses a fixed `random_state=42` for consistent results.
- **Output Storage**: Automatically organizes plots and submissions into dedicated directories.
- **Cross-Validation**: Performs Stratified K-Fold CV to estimate model performance (AUC).
- **Visualizations**: Generates plots in the `plots/` directory:
  - `feature_importance.png`: Top feature importances or coefficients.
  - `train_test_distribution_comparison.png`: Comparison of predicted probability distributions.
  - `train_roc_curve.png`: ROC curve for the training data.

## Project Structure

```text
.
├── data/               # Input CSV files (application_train.csv, application_test.csv)
├── plots/              # Generated visualization plots
├── submissions/        # Generated submission CSV files
├── main.py             # Main entry point for the project
├── pyproject.toml      # Poetry project configuration
├── poetry.lock         # Poetry lock file
├── requirements.txt    # Exported requirements (if available)
└── README.md           # Project documentation
```

## Testing

The project uses `pytest` for automated testing.

Run all tests:
```bash
poetry run python -m pytest
```

*(Note: Ensure you have added test files, e.g., `test_main.py`, as they are expected by the `pytest` command.)*

## License

This project is licensed under the MIT License.

## Environment Variables

No specific environment variables are required for basic execution.
