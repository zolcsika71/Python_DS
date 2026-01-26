# Home Credit Default Risk - Machine Learning Pipeline

## 1. Project Overview

The **Home Credit Default Risk** project is a production-grade machine learning pipeline designed to predict loan repayment difficulties. Unlike basic models that only use applicant demographics, this system integrates the full relational Kaggle dataset, encompassing historical credit behaviors, monthly balances, and installment patterns.

The primary objective is to provide **highly interpretable risk assessments** while maintaining robust performance in the face of data drift. This is achieved through a modular architecture that separates data orchestration, statistical mitigation, and model explainability.

### Key Functional Pillars:
- **Full Relational Integration**: Automated aggregation of multiple supplemental datasets (Bureau, Previous Applications, etc.) into a flat, feature-rich format.
- **Informed Drift Mitigation**: A proactive strategy that identifies covariate shift between training and production data, selectively dropping features that are unstable yet non-critical for prediction.
- **Probabilistic Reliability**: Integrated Platt scaling (via `CalibratedClassifierCV`) ensures that predicted probabilities reflect actual risk levels, which is crucial for financial decision-making.
- **Explainable AI (XAI)**: Utilizes SHAP (SHapley Additive exPlanations) to provide local and global interpretability, moving beyond simple "black-box" predictions.

---

## 2. Installation Instructions

### Prerequisites
- **Python**: v3.12 or higher.
- **Poetry**: Used for dependency management. If you don't have it, install it via `pip install poetry`.

### Setup Steps
1.  **Clone the Repository**:
    ```bash
    git clone <repository-url>
    cd Python_DS
    ```
2.  **Install Environment**:
    ```bash
    poetry install
    ```
3.  **Data Placement**:
    Create a `data/` directory in the root and place the Kaggle CSV files there. The pipeline expects:
    - `application_train.csv` & `application_test.csv` (Primary)
    - `bureau.csv`, `previous_application.csv`, `POS_CASH_balance.csv`, etc. (Supplemental)

---

## 3. Detailed Code Explanation

The project is structured into modular components within the `src/` directory, each responsible for a specific stage of the pipeline.

### A. Configuration (`src/config.py`)
**Purpose**: Centralizes all hyperparameters, file paths, and logging settings to ensure reproducibility.
- **Logic**: Uses Python `dataclasses` for structured configuration. It also implements a `ColorFormatter` for the logging system, which provides visual cues in the CLI (e.g., Green for success, Bold Blue for drift warnings).
- **Example Usage**:
```python
from src.config import CONFIG
# Access hyperparameters directly
lr = CONFIG.lgbm_params['learning_rate']
```

### B. Data Processing (`src/data_processing.py`)
**Purpose**: Handles the "heavy lifting" of data cleaning, joining, and feature engineering.
- **Functionality**:
    - `join_supplemental_data`: Performs complex many-to-one joins, aggregating historical data (Bureau, Previous Apps) using statistics like `mean`, `max`, and `sum`.
    - `fix_known_anomalies`: Identifies and flags data artifacts (e.g., the `365243` placeholder in `DAYS_EMPLOYED`).
- **Complex Logic - Informed Drift Mitigation**:
  The function `select_features_by_drift` implements a two-stage filter:
  1. It calculates the relative mean difference between train and test sets for all numerical features.
  2. If a feature exceeds the `drift_threshold`, it checks its importance (calculated via a pilot model). Only if the feature is **both drifted and low-importance** is it dropped. This preserves critical signals even if they have shifted.

### C. Modeling Pipeline (`src/modeling.py`)
**Purpose**: Encapsulates the Scikit-Learn pipeline and cross-validation logic.
- **Functionality**:
    - `build_pipeline`: Dynamically constructs a `ColumnTransformer`. It handles `SimpleImputer` and `StandardScaler` for numeric data, and `OneHotEncoder` for categorical data.
    - `try_build_model`: A factory function that initializes either `LGBMClassifier` or `LogisticRegression`, wrapped in `CalibratedClassifierCV`.
- **Reasoning**: We use a `Pipeline` object to prevent **data leakage** during cross-validation. Preprocessing parameters (like scaling means) are only learned from the training folds and then applied to the validation folds.

### D. Visualization & Explainability (`src/visualization.py`)
**Purpose**: Converts model outputs into actionable insights.
- **Features**:
    - **ROC Curves**: To evaluate the trade-off between sensitivity and specificity.
    - **SHAP Summary Plots**: Provides a global view of which features drive the model's decisions and in which direction.
- **Example Usage**:
```python
# Usage within a script:
# from src.visualization import plot_shap_summary
# plot_shap_summary(clf, x_sample, "plots/shap_summary.png")
```

### E. Orchestrator (`src/orchestrator.py`)
**Purpose**: The "brain" of the project that connects all modules into a linear workflow.
- **Chain-of-Thought**:
  1. **Load**: Ingests all CSVs.
  2. **Pilot Fit**: Performs a quick training pass to determine feature importance for the drift mitigation step.
  3. **Refine**: Drops problematic features and applies final engineering.
  4. **Cross-Validate**: Estimates out-of-sample performance across multiple folds.
  5. **Final Train & Plot**: Trains on the full dataset and generates all diagnostic visualizations.

---

## 4. Usage Examples

### Full Pipeline Execution
The easiest way to run the project is via `main.py`:
```bash
# Recommended: LightGBM with 5-fold CV
poetry run python main.py --prefer-lightgbm --folds 5
```

### High-Risk Customer Analysis
After running the pipeline, you can perform a focused analysis on the most "at-risk" applicants:
```bash
poetry run python src/top_10_analysis.py
```
This script identifies the 10 customers with probabilities closest to 1.0 and generates a dedicated plot in the `plots/` folder.

---

## 5. Contribution Guidelines

We follow a strict "Clean Code" philosophy to ensure the pipeline remains maintainable:

- **Modularity**: Every new feature should be a pure function in `data_processing.py` or a distinct step in the `Pipeline`.
- **Testing**: Add a corresponding test in `tests/` for any new logic. Run existing tests with:
  ```bash
  poetry run python tests/test_data_processing.py
  ```
- **Documentation**: Use Google-style docstrings.
- **Style**: Adhere to PEP 8. Use the centralized `logger` for all console output.

---

## 6. Changelog & Versioning

The project follows semantic versioning. 
- **Current Version**: `v1.5.1`
- **Latest Change**: Integrated BOLD styling for CLI warnings to improve accessibility and visibility of data drift alerts.


MIT license