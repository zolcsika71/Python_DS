# Home Credit Default Risk - Machine Learning Pipeline

This project implements a modular machine learning pipeline for the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) Kaggle competition. It is designed to predict whether an applicant will have difficulty repaying a loan.

## 🚀 Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for data processing, modeling, and visualization.
- **Robust Preprocessing**: Automatic handling of categorical (One-Hot Encoding) and numerical features (Imputation, Scaling).
- **Anomaly Detection**: Specialized handling for known dataset anomalies (e.g., `DAYS_EMPLOYED` placeholders).
- **Flexible Modeling**: Supports both `LogisticRegression` as a stable baseline and `LightGBM` for high performance.
- **Evaluation & Visualization**: Comprehensive ROC-AUC analysis, feature importance plots, and prediction distribution comparisons.
- **Enhanced Logging**: Console output features color-coded levels, with `INFO` messages highlighted in green for better readability.
- **Automated Workflow**: End-to-end execution from raw data to submission-ready CSV files.

## 📁 Project Structure

```text
.
├── main.py                # Thin CLI entry point
├── src/
│   ├── config.py          # Centralized configuration and logging
│   ├── data_processing.py # Data loading and cleaning logic
│   ├── modeling.py        # Pipeline building and model definitions
│   ├── visualization.py   # Plotting and evaluation functions
│   └── orchestrator.py    # Workflow orchestration
├── data/                  # Input datasets (application_train.csv, application_test.csv)
├── plots/                 # Generated visualizations (ROC, Importance, etc.)
├── submissions/           # Timestamped submission CSVs
├── test_data_processing.py # Unit tests for processing logic
├── pyproject.toml         # Poetry dependency configuration
└── README.md              # Project documentation
```

## 🛠️ Setup

### Prerequisites
- Python >= 3.12
- [Poetry](https://python-poetry.org/)

### Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   poetry install
   ```

### Data Placement
Download the competition data from Kaggle and place the following files in the `data/` directory:
- `application_train.csv`
- `application_test.csv`

## 💻 Usage

### Running the Pipeline
The `main.py` script handles the entire workflow:
```bash
poetry run python main.py [OPTIONS]
```

### Options
- `--data-dir`: Custom path to data directory (default: `data`).
- `--folds`: Number of cross-validation folds (default: `3`).
- `--prefer-lightgbm`: Use LightGBM if installed (recommended for better results).
- `--out`: Specific output path for the submission CSV.

### Examples
**Standard Run (Logistic Regression):**
```bash
poetry run python main.py
```

**High Performance Run (LightGBM + 5-fold CV):**
```bash
poetry run python main.py --prefer-lightgbm --folds 5
```

## 🧪 Testing
The project uses `pytest`. To run the tests:
```bash
poetry run python -m pytest
```
Note: If `pytest` is not installed in your environment, you can add it via `poetry add --group dev pytest`.

## 📊 Visualizations
After execution, check the `plots/` directory for:
- `feature_importance.png`: Visualizes the most impactful features.
- `train_roc_curve.png`: Shows the model's performance on the training set.
- `train_test_distribution_comparison.png`: Ensures consistency between train and test predictions.

## ⚖️ License
This project is licensed under the MIT License.
