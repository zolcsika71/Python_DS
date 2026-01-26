# Home Credit Default Risk - Machine Learning Pipeline

This project provides a comprehensive, modular machine learning pipeline designed to solve the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) Kaggle competition. The goal is to predict the probability that an applicant will encounter difficulties repaying a loan, utilizing both traditional and alternative data sources.

---

## 1. Project Overview

The Home Credit Default Risk pipeline is built with a focus on **modularity**, **reproducibility**, and **maintainability**. It automates the entire machine learning lifecycle, from data ingestion and cleaning to model training, cross-validation, and the generation of submission-ready artifacts.

### Main Features:
- **Dual Model Support**: Easily switch between a robust `LogisticRegression` baseline and tuned `LightGBM` (with early stopping and optimized hyperparameters). Now includes **Probability Calibration** (Platt scaling) as a default for both.
- **Automated Preprocessing**: Handles missing values and scales features automatically based on data types using an integrated `ColumnTransformer` pipeline.
- **Anomaly Handling**: Specifically addresses known data issues like the `DAYS_EMPLOYED` anomaly and flags them for the model.
- **Engineered Ratios**: Includes credit-to-income, annuity-to-income, goods-to-credit, and employed-to-birth ratios, plus automated aggregation of all supplemental datasets.
- **Informed Drift Mitigation**: Detects data drift (covariate shift) and automatically drops features that are statistically shifted between sets but have low predictive importance.
- **Advanced Visualization & Explainability**: Generates ROC curves, feature importance plots, distribution comparisons, and **SHAP summary plots** for model interpretability.
- **Risk Analysis**: A specialized "Top 10" analysis identifies and visualizes customers with the highest predicted risk.
- **Professional Logging**: Uses a custom color-coded logging system (green for success/info) for better visibility.
- **Full Relational Integration**: Automatically joins and aggregates `bureau`, `previous_applications`, `POS_CASH`, `installments`, and `credit_card_balance` datasets.

---

## 2. Installation Instructions

### Prerequisites
- **Python**: version 3.12 or higher.
- **Poetry**: Dependency management and packaging tool.

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
3.  **Data Requirements**:
    The project utilizes the full relational structure of the Kaggle dataset. Place the following files in the `data/` directory:
    - `application_train.csv` (Required)
    - `application_test.csv` (Required)
    - `bureau.csv` (Optional, for historical credit data)
    - `bureau_balance.csv` (Optional, for monthly bureau status)
    - `previous_application.csv` (Optional, for internal loan history)
    - `POS_CASH_balance.csv` (Optional, for monthly POS/Cash loan status)
    - `installments_payments.csv` (Optional, for repayment history)
    - `credit_card_balance.csv` (Optional, for monthly credit card status)

---

## 3. Usage Guidelines

The pipeline is primarily controlled through `main.py`, but also includes standalone scripts for specific tasks.

### Running the Full Pipeline
Use `main.py` to execute the end-to-end workflow (loading, cleaning, training, evaluation, and submission generation).

```bash
# Basic run with 3-fold CV (Logistic Regression default)
poetry run python main.py

# High-performance run with LightGBM and 5-fold CV
poetry run python main.py --prefer-lightgbm --folds 5

# Customizing output path
poetry run python main.py --out submissions/my_custom_submission.csv
```

### Standalone Scripts
Located in the `src/` directory:
- **`src/top_10_analysis.py`**: Runs a standalone analysis on the latest submission to identify high-risk cases.
  ```bash
  poetry run python src/top_10_analysis.py
  ```
- **`src/process_target.py`**: A utility to post-process submissions (e.g., capping probabilities).
  ```bash
  poetry run python src/process_target.py
  ```

---

## 4. Directory Structure

- **`main.py`**: Application entry point.
- **`src/`**: Core logic (config, processing, modeling, visualization, orchestrator, and scripts).
- **`tests/`**: Unit tests for all major components.
- **`data/`**: Input CSV files.
- **`plots/`**: Generated visualizations (ROC, Importance, Top 10).
- **`submissions/`**: Output predictions and analysis CSVs.
- **`docs/`**: Documentation and presentation materials.

---

## 5. Contributing

We welcome contributions! To maintain code quality, please follow these guidelines:

### Coding Standards
- **PEP 8**: Follow standard Python style conventions.
- **Type Hinting**: Use type hints for all function signatures.
- **Logging**: Use the centralized `logger` from `src.config` instead of `print()`.
- **Modularity**: Keep functions focused and well-documented with Google-style docstrings.

### Pull Request Process
1. Fork the repo and create a feature branch.
2. Implement your changes and add tests if applicable.
3. Ensure all tests pass (`poetry run python -m pytest` or manual execution of scripts in `tests/`).
4. Submit a Pull Request with a clear description of the changes.

---

## 6. Testing Instructions

The project uses `pytest` for automated testing.

### Running All Tests
```bash
# If pytest is installed in the environment
poetry run pytest

# Alternatively, run individual test scripts
poetry run python tests/test_data_processing.py
poetry run python tests/test_modeling.py
poetry run python tests/test_config.py
poetry run python tests/test_visualization.py
poetry run python tests/test_orchestrator.py
```

---

## 7. License Information

This project is released under the **MIT License**. See the [LICENSE](LICENSE) file for the full text (if available) or visit [opensource.org/licenses/MIT](https://opensource.org/licenses/MIT).

---

## 8. Changelog

### [v1.5.1] - 2026-01-26
- **Dependency Fix**: Added missing `shap` and `numba` dependencies to ensure the pipeline can generate SHAP-based model explanations without errors.
- **Documentation Update**: Refreshed README.md to reflect the latest pipeline features and automated drift mitigation strategies.

### [v1.5.0] - 2026-01-23
- **Automated Drift Mitigation**: Implemented informed feature selection to address data drift. The pipeline now automatically identifies drifted features and drops those with low predictive importance, improving model robustness against distribution shifts.
- **Enhanced Drift Reporting**: Updated drift detection to sort warnings by severity (relative difference) and provide clearer logs.

### [v1.4.0] - 2026-01-23
- **Full Relational Integration**: Refactored the data processing pipeline to automatically ingest and aggregate all available Kaggle data files including **`bureau_balance.csv`**, `bureau.csv`, `previous_applications`, `POS_CASH`, `installments`, and `credit_card_balance`.
- **Dynamic Feature Engineering**: Implemented robust aggregation logic (mean, max, sum, count) for one-to-many relational tables, increasing the feature set from 122 to 180+.
- **Data Drift Detection**: Integrated automated drift monitoring, identifying significant shifts in 50 features.
- **Performance Improvement**: Mean AUC increased significantly (from ~0.745 to ~0.761) due to the inclusion of historical behavioral data.

### [v1.3.0] - 2026-01-23
- **Probability Calibration**: Integrated `CalibratedClassifierCV` for better probability estimates.
- **Tuned LightGBM**: Optimized hyperparameters and added early stopping support.
- **Feature Engineering**: Added financial ratios (credit-to-income, annuity-to-income, etc.).
- **Data Validation**: Added automated schema checks and data drift detection.
- **Explainability**: Integrated SHAP for feature-level model interpretation.

### [v1.2.0] - 2026-01-22
- **Refactored Directory Structure**: Migrated standalone scripts from `src/scripts/` to `src/` for a flatter, more efficient structure.
- **Enhanced Orchestration**: Integrated "Top 10" analysis directly into the main pipeline.
- **Visualization Update**: Centralized all plotting logic into `src/visualization.py`.
- **Improved Automation**: Scripts now automatically detect the latest submission files.
- **Expanded Test Suite**: Added comprehensive tests for configuration, visualization, and orchestration.

### [v1.1.0] - 2026-01-21
- Initial major refactor: Introduced `ModelConfig` and modularized data processing/modeling.
- Added support for LightGBM with Logistic Regression fallback.

---
*Last Updated: 2026-01-26*

---

## 9. Risk Assessment & Data Drift Report

### What is Data Drift?
Data drift (specifically covariate shift) occurs when the statistical properties of the input data change between the training set and the test set. For example:
- **`FLAG_EMAIL` (1.87 drift)**: Indicates a 187% difference in mean, suggesting a change in how contact information was collected.
- **`INSTAL_PAYMENT_DIFF_SUM` (1.33 drift)**: Suggests the test set contains applicants with more volatile payment histories.

### Why It Matters
If a model relies on features that have drifted, its learned patterns may no longer apply to the new data, leading to **performance degradation** on the leaderboard.

### Automated Mitigation (v1.5.0)
The pipeline proactively addresses this using an **Informed Feature Selection** strategy:
1. **Detection**: Compares means between train and test sets to identify shifted features.
2. **Importance Evaluation**: Trains a pilot model to estimate the predictive power of each feature.
3. **Intelligent Dropping**: 
   - **High Drift + Low Importance**: Automatically dropped to reduce noise and risk.
   - **High Drift + High Importance**: Kept (e.g., `AMT_CREDIT`), as their predictive value outweighs the distribution shift risk.

### Summary of Findings (2026-01-26)
- **Detected Drift**: ~50 columns identified with significant shifts.
- **Severity**: **Moderate**. While the system is robust, these shifts require the automated mitigation currently in place.
- **Action Taken**: The pipeline successfully identified and dropped problematic features (e.g., administrative flags and low-impact counts) while preserving high-value predictors.
