# Home Credit Default Risk - Machine Learning Pipeline

This project provides a comprehensive, modular machine learning pipeline designed to solve the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) Kaggle competition. The goal is to predict the probability that an applicant will encounter difficulties repaying a loan, utilizing both traditional and alternative data sources.

---

## 1. Project Overview

The Home Credit Default Risk pipeline is built with a focus on **modularity**, **reproducibility**, and **maintainability**. It automates the entire machine learning lifecycle, from data ingestion and cleaning to model training, cross-validation, and the generation of submission-ready artifacts.

### Main Features:
- **Automated Preprocessing**: Handles missing values and scales features automatically based on data types.
- **Anomaly Handling**: Specifically addresses known data issues like the `DAYS_EMPLOYED` anomaly.
- **Engineered Ratios**: Includes credit-to-income, annuity-to-income, goods-to-credit, and employed-to-birth ratios.
- **Data Validation**: Automated schema validation and relative mean drift detection between train and test sets.
- **Dual Model Support**: Easily switch between a robust `LogisticRegression` baseline and tuned `LightGBM` (with early stopping and optimized hyperparameters).
- **Probability Calibration**: Uses Platt scaling (`CalibratedClassifierCV`) to ensure well-calibrated risk estimates.
- **Advanced Visualization & Explainability**: Generates ROC curves, feature importance plots, distribution comparisons, and **SHAP summary plots** for model interpretability.
- **Top 10 Analysis**: A specialized analysis that identifies and visualizes customers with the highest predicted risk (probabilities closest to 1.0).
- **Professional Logging**: Uses a custom color-coded logging system for better visibility of the pipeline's progress.

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
    Place the Kaggle competition files (`application_train.csv` and `application_test.csv`) in the `data/` directory.

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
*Last Updated: 2026-01-23*
