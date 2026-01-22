# Home Credit Default Risk - Machine Learning Pipeline

This project provides a comprehensive, modular machine learning pipeline designed to solve the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) Kaggle competition. The goal is to predict the probability that an applicant will encounter difficulties repaying a loan, utilizing both traditional and alternative data sources.

---

## 1. Project Overview

The Home Credit Default Risk pipeline is built with a focus on **modularity**, **reproducibility**, and **maintainability**. It automates the entire machine learning lifecycle, from data ingestion and cleaning to model training, cross-validation, and the generation of submission-ready artifacts.

### Main Features:
- **Automated Preprocessing**: Handles missing values and scales features automatically based on data types.
- **Anomaly Handling**: Specifically addresses known data issues like the `DAYS_EMPLOYED` anomaly.
- **Dual Model Support**: Easily switch between a robust `LogisticRegression` baseline and high-performance `LightGBM`.
- **Advanced Visualization**: Generates ROC curves, feature importance plots, and distribution comparisons. It also performs a specialized "Top 10" analysis to identify customers with the highest predicted risk.
- **Professional Logging**: Uses a custom color-coded logging system for better visibility of the pipeline's progress.

### Technologies Used:
- **Python 3.12+**
- **Poetry** (Dependency Management)
- **Scikit-learn** (Preprocessing and Modeling)
- **LightGBM** (Gradient Boosting)
- **Pandas/NumPy** (Data Manipulation)
- **Matplotlib/Seaborn** (Visualization)

---

## 2. Directory Structure

The codebase is organized as follows:

- **`main.py`**: The primary entry point for the application. It handles Command Line Interface (CLI) arguments and triggers the orchestration logic.
- **`src/`**: Contains the core logic of the pipeline, divided into specialized modules.
    - **`config.py`**: Centralized configuration using structured dataclasses (`ModelConfig`, `PathConfig`).
    - **`data_processing.py`**: Functions for reading raw data and performing initial cleaning/anomaly fixing.
    - **`modeling.py`**: Logic for building Scikit-learn pipelines including modeling steps and cross-validation.
    - **`visualization.py`**: Utilities for creating diagnostic plots (ROC curves, importance charts, etc.).
    - **`orchestrator.py`**: Defines the high-level workflow that connects all other modules, including the top 10 analysis logic.
    - **`scripts/`**: Useful post-processing scripts.
        - **`process_target.py`**: Post-processes submission files by identifying and capping the highest predictions.
        - **`top_10_analysis.py`**: Analyzes and visualizes the top 10 TARGET values closest to 1 from a submission file.
- **`tests/`**: Unit tests to ensure the reliability of core components.
    - **`test_config.py`**: Validates configuration loading and directory management.
    - **`test_visualization.py`**: Ensures all plotting utilities function correctly.
    - **`test_orchestrator.py`**: Tests high-level workflow components like the top 10 analysis.
    - **`test_data_processing.py`**: Verifies anomaly fixing logic.
    - **`test_modeling.py`**: Validates pipeline construction and model fitting.
- **`docs/`**: Project documentation and presentations.
    - **`Home_Credit_Default_Risk_v4.pptx`**: Project presentation.
    - **`PPTX_UPDATE_NOTES.md`**: Detailed notes on recent changes to be reflected in the presentation.
- **`data/`**: Directory for input CSV files (`application_train.csv`, `application_test.csv`).
- **`plots/`**: Automatically generated directory where training visualizations and analysis plots (e.g., `top_10_targets_closest_to_1.png`) are saved.
- **`submissions/`**: Automatically generated directory for output CSV files and analysis results (e.g., `top_10_closest_targets.csv`).

---

## 3. Module/Component Descriptions

### `main.py`
The primary entry point for the application. It handles Command Line Interface (CLI) arguments and triggers the orchestration logic.
- **Key Functions**:
    - `main()`: Uses `argparse` to handle user inputs such as data directories, number of CV folds, model preference, and output paths.
- **Interactions**: Acts as the user interface, passing configuration to `src/orchestrator.py`.

### `src/config.py`
This module centralizes all project-wide settings to ensure consistency across different execution environments.
- **Key Functions/Classes**:
    - `ColorFormatter`: A custom logging formatter that colors `INFO` messages green for better readability in the terminal.
    - `ModelConfig`: A frozen dataclass that stores hyperparameters for both Logistic Regression and LightGBM models.
    - `setup_directories()`: Ensures that `plots/` and `submissions/` folders exist before the pipeline attempts to write to them.
- **Interactions**: Imported by almost every other module to access the global `logger` and the `CONFIG` object.

### `src/data_processing.py`
Responsible for the "Extract and Transform" part of the ETL process.
- **Key Functions**:
    - `load_data(data_dir)`: Reads the training and testing datasets from CSV files.
    - `fix_known_anomalies(df)`: Implements domain-specific cleaning logic, such as handling the "1000-year" employment anomaly.
- **Interactions**: Called early in the `orchestrator.py` workflow to prepare data for modeling.

### `src/modeling.py`
Contains the structural definitions of the machine learning models and the cross-validation logic.
- **Key Functions**:
    - `build_pipeline(cat_cols, num_cols)`: Creates a `ColumnTransformer` that imputes and scales data based on whether a column is categorical or numerical.
    - `try_build_model(prefer_lightgbm)`: A factory function that attempts to instantiate `LGBMClassifier` but falls back to `LogisticRegression` if LightGBM is unavailable.
    - `cross_validate_auc(x, y, folds)`: Performs Stratified K-Fold cross-validation and logs the Mean AUC score.
- **Interactions**: Receives cleaned data from the orchestrator and provides trained pipelines back to it.

### `src/visualization.py`
Provides visual evidence of model performance and data health.
- **Key Functions**:
    - `plot_feature_importance(clf)`: Extracts coefficients or importance scores from the trained model and creates a bar chart.
    - `plot_roc_curve(y_true, y_probes)`: Generates the ROC curve and calculates the AUC for the training set.
    - `plot_train_test_distribution(train_proba, test_proba)`: Overlays histograms of predictions to check for model stability.
- **Interactions**: Called by the orchestrator after the final model training is complete.

### `src/orchestrator.py`
The central workflow coordinator that connects all components into a single pipeline.
- **Key Functions**:
    - `run_pipeline(...)`: Orchestrates the sequence of events: directory setup, data loading, anomaly fixing, cross-validation, final training, visualization, and submission generation.
    - `analyze_top_10_targets(submission_df, config)`: Identifies the 10 predictions closest to 1.0, saves a visualization to `plots/`, and exports the data to `submissions/top_10_closest_targets.csv`.
- **Interactions**: Imports and invokes functionality from all other `src/` modules.

### `src/scripts/`
Useful post-processing scripts located within the `src` directory for better organization.
- **Key Scripts**:
    - `process_target.py`: Post-processes submission files by identifying and capping the highest predictions.
    - `top_10_analysis.py`: Analyzes and visualizes the top 10 TARGET values closest to 1 from a submission file.
- **Interactions**: These scripts are designed to be run standalone and import core logic from `src`.

### `tests/`
Unit tests designed to ensure the stability of the core logic.
- **Key Tests**:
    - `test_config.py`: Validates that directory setup and configuration defaults are correct.
    - `test_visualization.py`: Confirms that plots are correctly generated and saved.
    - `test_orchestrator.py`: Tests the top 10 analysis and core orchestration logic.
    - `test_data_processing.py`: Verifies that the employment anomaly flag is correctly created.
    - `test_modeling.py`: Confirms that the `ColumnTransformer` correctly processes different data types.
- **Interactions**: Standalone scripts that import `src` modules to validate their behavior via `pytest`.

---

## 4. Code Snippets & Explanations

### Custom Logging for Better UX
The pipeline uses ANSI escape codes to make progress messages stand out.

```python
import logging

class ColorFormatter(logging.Formatter):
    GREEN = "\033[92m"
    RESET = "\033[0m"

    def format(self, record):
        if record.levelno == logging.INFO:
            formatted_msg = super().format(record)
            return f"{self.GREEN}{formatted_msg}{self.RESET}"
        return super().format(record)
```
*Explanation*: This snippet from `src/config.py` wraps `INFO` level logs in green color. This allows users to quickly distinguish between standard progress updates and warnings or errors.

### Handling Data Anomalies
The dataset contains a known placeholder for employment length that can mislead models.

```python
import pandas as pd
import numpy as np

def fix_known_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    if "DAYS_EMPLOYED" in df.columns:
        anom_val = 365243 # Placeholder for ~1000 years
        df = df.copy()
        # Create a boolean flag for the anomaly
        df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == anom_val).astype(np.int8)
        # Replace with NaN so the Imputer can handle it appropriately
        df.loc[df["DAYS_EMPLOYED"] == anom_val, "DAYS_EMPLOYED"] = np.nan
    return df
```
*Explanation*: Instead of simply deleting the anomalous rows, we capture the fact that the data was missing (via the `ANOM` flag) and then use statistical imputation for the numerical value. This preserves information while cleaning the feature.

### Dynamic Pipeline Construction
The pipeline is built dynamically based on the data types of the input columns and includes the model directly. This ensures that preprocessing, cleaning, and modeling are all handled within a single object, improving consistency and reducing errors.

### Orchestrating the Workflow
The `run_pipeline` function in `src/orchestrator.py` manages the high-level execution sequence:
1.  **Environment Setup**: Ensures output directories exist.
2.  **Data Ingestion**: Loads training and testing data.
3.  **Data Cleaning**: Fixes known anomalies like the `DAYS_EMPLOYED` placeholder.
4.  **Model Validation**: Performs Stratified K-Fold cross-validation to estimate performance (AUC).
5.  **Final Training**: Trains the chosen model on the full dataset.
6.  **Analysis & Artifacts**: Generates diagnostic plots, identified the top 10 TARGET values closest to 1, and writes the submission file.

---

## 5. Installation and Usage Instructions

### Prerequisites
- Python 3.12+
- Poetry (Package Manager)

### Local Setup
1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd Python_DS
    ```
2.  **Install dependencies**:
    ```bash
    poetry install
    ```

### Data Placement
Place the Kaggle competition files (`application_train.csv` and `application_test.csv`) in the `data/` directory.

### Running the Application
The pipeline is launched via `main.py`. You can configure its behavior using command-line arguments:
```bash
# Standard run using Logistic Regression (default)
poetry run python main.py

# High-performance run using LightGBM and 5-fold CV
poetry run python main.py --prefer-lightgbm --folds 5
```

---

## 6. Examples of Common Use Cases

### Scenario A: Fast Model Validation
To quickly check if the pipeline is working on a new machine, run a 2-fold cross-validation:
```bash
poetry run python main.py --folds 2
```

### Scenario B: Generating a Submission for Kaggle
To generate the most accurate submission possible (requires `lightgbm` installed):
```bash
poetry run python main.py --prefer-lightgbm --folds 5 --out submissions/my_best_submission.csv
```

---

## 7. Contribution Guidelines

We welcome contributions to improve the pipeline!

### Coding Standards
- **PEP 8**: All Python code should follow PEP 8 style guidelines.
- **Type Hinting**: Use type hints for function arguments and return values where possible.
- **Logging**: Do not use `print()`. Use the centralized `logger` from `src.config`.

### Testing Requirements
- If you add a new feature, you **must** add a corresponding test in the `tests/` directory.
- Ensure all tests pass before submitting a PR:
  ```bash
  poetry run python -m pytest
  ```

### How to Submit a Pull Request
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/improvement`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature/improvement`).
5. Open a Pull Request on GitHub.

---

## 8. Target Variable Identification

- **Project Context**: The primary objective is to predict whether an applicant will encounter difficulties repaying a loan, helping Home Credit broaden financial inclusion for the unbanked population by leveraging alternative data.
- **Key Metrics**: 
    - *Quantitative*: `AMT_INCOME_TOTAL`, `AMT_CREDIT`, `DAYS_EMPLOYED`, and `EXT_SOURCE` scores.
    - *Qualitative*: `NAME_EDUCATION_TYPE`, `OCCUPATION_TYPE`, and family status.
- **Selection Process**: The target variable is **`TARGET`**. It is a binary indicator (1 for payment difficulties, 0 otherwise). This is the most important variable as it directly measures the financial risk. The pipeline specifically analyzes the top 10 values closest to 1.0 to provide insight into the most high-risk cases identified by the model.
- **Examples**: Similar to the LendingClub Loan Analysis or Fannie Mae Mortgage Default Prediction, where a binary status allows for identifying high-risk patterns to improve portfolio health.

---
*License: This project is licensed under the MIT License.*
