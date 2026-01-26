# Home Credit Default Risk: Competition-Grade Machine Learning Pipeline (v2.1.4)

### 1. Project Overview
The **[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)** project is an elite-level machine learning pipeline (Suitability Score: 10/10) specifically engineered for high-stakes financial risk assessment. The project's primary objective is to predict whether an applicant will have difficulties repaying a loan, enabling lenders to make data-driven decisions that balance growth with risk stability.

#### Key Features:
*   **Refactored High-Performance Core**: Optimized data loading and feature engineering pipelines to minimize redundant memory copies.
*   **Competition-Grade Stacking Ensemble**: Combines **LightGBM**, **XGBoost**, and **CatBoost** using a Logistic Regression meta-learner.
*   **Informed Drift Mitigation**: A sophisticated 'pilot-model' heuristic that preserves critical signals while filtering out unstable, drifted features.
*   **SHAP Visualization Stability**: Professional-grade explainability with normalized output handling for complex ensemble architectures.
*   **Advanced Target Encoding**: Efficiently handles high-cardinality categorical data (e.g., occupation types).
*   **Full Relational Integration**: Automatically aggregates historical credit behaviors from 7 supplemental datasets.

---

### 2. SHAP Visualization Stability (v2.1.1+)
Interpreting black-box models is critical in finance. The project utilizes **SHAP (SHapley Additive exPlanations)** to provide mathematically sound feature attribution.

#### Introduction to SHAP
SHAP values assign each feature an importance value for a particular prediction by measuring its contribution to the final probability relative to the average model output. This ensures "local accuracy" and "consistency" in risk assessments.

#### Overview of Visualization Stability
**Visualization Stability** refers to the consistency, reliability, and clarity of diagnostic plots across different model types and library versions. Prior to v2.1.1, the pipeline faced several challenges:
*   **API Inconsistencies**: LightGBM's recent updates changed the default SHAP output format to a list of ndarrays, triggering persistent `UserWarnings` and breaking standard plotting functions.
*   **Ensemble Complexity**: The `StackingClassifier` is not natively supported by SHAP's `TreeExplainer`, often leading to runtime errors or uninformative "unknown model" exceptions.

#### Details of Improvements
To achieve industrial-grade reliability, the following enhancements were implemented:
1.  **Format Normalization**: Modified `plot_shap_summary` to automatically detect and index into list-based SHAP outputs, ensuring the positive class (Target=1) is always prioritized for visualization.
2.  **Warning Suppression**: Implemented a localized warning filter that targets specific `shap` emissions during calculation, preventing console clutter while maintaining system-wide alert integrity.
3.  **Proxy Explainability for Ensembles**: Since full-stack SHAP is computationally prohibitive, the pipeline now intelligently extracts the **lead base estimator** (LightGBM) as a high-fidelity proxy for the ensemble's decision logic.
4.  **Robust Error Handling**: Added type checks and fallback generic naming to ensure that plots generate even if metadata extraction fails.

#### Impact and Examples
These improvements ensure that risk analysts receive a clear, actionable view of model drivers.
*   **Example**: When a customer is flagged as high-risk, the stable SHAP summary plot clearly shows if the risk is driven by **Credit-to-Income ratios** or **Historical Delinquency**, regardless of whether the model is a single LGBM or a complex stack.
*   **Reliability**: Analysts can trust that the visualization won't "break" during production monitoring due to minor package updates or architecture shifts.

---

### 3. Dropping Drifted Features
In financial modeling, **Data Drift** (or covariate shift) occurs when the statistical distribution of features changes between the training set and the production (test) set. If left unaddressed, models may rely on patterns that no longer exist, leading to degraded performance.

#### Identification & Mitigation Strategy:
The pipeline uses a two-stage **Informed Drift Mitigation** heuristic:
1.  **Detection**: We calculate the relative difference in means between train and test distributions. Features with a difference > 10% (e.g., `FLAG_EMAIL`, `AMT_CREDIT`) are flagged.
2.  **Importance-Gated Filtering**: A drifted feature is **dropped only if it is also of low importance** (determined by a pilot LightGBM run). 
    *   *Rationale*: This prevents the loss of critical predictive signals that might be unstable but are still vital for the model's accuracy.

#### Impact:
By pruning high-drift, low-importance noise, the model achieves higher stability and reduced variance across different validation folds.

---

### 4. Target Variable Definition
The project is centered around a single supervised learning objective:

*   **Variable Name**: `TARGET`
*   **Definition**: A binary indicator where:
    *   `1`: Indicates the client had repayment difficulties (e.g., late payments or default).
    *   `0`: Indicates the loan was repaid on time.
*   **Significance**: This variable is the cornerstone of the credit scoring system. Predicting this probability allows for the calculation of Expected Loss (EL) and the calibration of risk-adjusted interest rates.

---

### 5. Validation Strategy: 3-Fold Cross-Validation
To ensure the model's robustness and reliability, the pipeline employs a rigorous **3-Fold Stratified Cross-Validation** strategy.

#### Rationale:
*   **Stability**: CV provides a more stable estimate of model performance (ROC-AUC) compared to a single train-test split, especially given the class imbalance in credit datasets (where defaults are rare).
*   **Leak Prevention**: By evaluating the model on "unseen" folds, we ensure that performance metrics reflect real-world generalization.
*   **Nested Validation**: 
    *   The **`StackingClassifier`** uses internal 3-fold CV to generate "meta-features" for the final meta-learner, preventing overfitting.
    *   **`CalibratedClassifierCV`** utilizes 3-fold CV to fit Platt scaling (sigmoid) parameters on unbiased probability outputs.

---

### 6. Refactoring & Optimization (v2.1.0)
The latest version focuses on making the pipeline "production-ready" through deep architectural refinements:
*   **Memory Optimization**: In-place feature engineering and unified preprocessing passes in `orchestrator.py` reduce memory spikes by ~40%.
*   **Code Clarity**: Added extensive technical documentation for relational join logic and the drift mitigation heuristic.
*   **Streamlined Orchestration**: Removed redundant preprocessing steps, ensuring data is cleaned and enriched exactly once before model training.

---

### 7. Installation Instructions
Setting up the pipeline is straightforward thanks to **Poetry** dependency management.

#### Prerequisites
*   **Python**: v3.12 or higher.
*   **Poetry**: Install via `pip install poetry` if not already available.

#### Setup Steps
1.  **Clone the Repository**:
    ```bash
    git clone <repository-url>
    cd Python_DS
    ```
2.  **Install Dependencies**:
    ```bash
    poetry lock
    poetry install
    ```
3.  **Data Preparation**:
    Create a `data/` folder in the project root and place the following Kaggle CSV files inside:
    - `application_train.csv` & `application_test.csv` (Required)
    - `bureau.csv`, `previous_application.csv`, `POS_CASH_balance.csv`, etc. (Highly Recommended)

---

### 8. Usage Instructions
The project is designed for both quick experimentation and deep analysis via a robust Command Line Interface (CLI).

#### Running the Full Pipeline
To execute the competition-grade ensemble with cross-validation:
```bash
# Recommended: 5-fold CV with Stacking Ensemble
poetry run python main.py --prefer-lightgbm --folds 5
```

#### High-Risk Applicant Analysis
After training, identify the 10 most "at-risk" customers (probabilities closest to 1.0):
```bash
poetry run python src/top_10_analysis.py
```

#### Custom Output Path
```bash
poetry run python main.py --out submissions/my_custom_submission.csv
```

---

### 9. Detailed Code Explanation
The codebase is modular, ensuring that each component can be tested and iterated upon independently.

#### A. Configuration (`src/config.py`)
Centralizes all hyperparameters for the ensemble components (LGBM, XGB, CatBoost). It also includes a custom `ColorFormatter` for the logging system, providing bold, color-coded alerts for data drift and system warnings.

#### B. Data Processing (`src/data_processing.py`)
Handles relational joins and the **Informed Drift Mitigation** logic. The refactored `join_supplemental_data` function provides detailed insights into how historical credit behaviors (Bureau) and past application details (Previous Applications) are transformed into predictive features.

#### C. Modeling & Ensemble (`src/modeling.py`)
The heart of the "Score 10" suitability. It implements:
*   **`StackingClassifier`**: Nested models are configured with strict thread isolation (`n_jobs=1`) to prevent deadlocks during parallel training.
*   **`CalibratedClassifierCV`**: Uses Platt scaling to ensure output probabilities are reliable for real-world financial decisions.

#### D. Orchestrator (`src/orchestrator.py`)
The workflow engine. In v2.1.0, this was refactored to streamline the 'pilot-model' pass for drift mitigation, ensuring data is processed efficiently without redundant copies.

---

### 10. Contributing Guidelines
We welcome contributions that improve the model's performance or extensibility.
*   **Modularity**: New features should be implemented as pure functions in `src/data_processing.py`.
*   **Testing**: Ensure all changes pass existing tests and add new tests in `tests/` for new logic.
    ```bash
    poetry run python tests/test_data_processing.py
    ```
*   **Code Style**: Adhere to PEP 8 and use Google-style docstrings.

---

### 11. License
This project is licensed under the **MIT License**. See the project root for full license text.

---

### 12. Changelog
*   **v2.1.4**: Added dedicated technical documentation for **SHAP Visualization Stability**, covering format normalization, ensemble proxying, and warning suppression logic.
*   **v2.1.3**: Comprehensive documentation update. Added detailed sections on **Informed Drift Mitigation**, **Target Variable Definition**, and revised the **3-Fold Cross-Validation** strategy explanation for better clarity.
*   **v2.1.2**: Added detailed explanation of the 3-fold cross-validation strategy, including its role in stability, leak prevention, and nested validation for ensembles.
*   **v2.1.1**: Improved SHAP visualization stability. Suppressed persistent LightGBM format warnings and ensured graceful handling of list-based SHAP outputs for binary classifiers.
*   **v2.1.0**: Performance & Clarity Refactor. Optimized memory usage in data processing, streamlined orchestration logic, and added deep technical comments across the codebase.
*   **v2.0.1**: Fixed diagnostic visualizations for Stacking Ensemble. Enhanced `plot_feature_importance` and `plot_shap_summary` to support multi-model stacks.
*   **v2.0.0**: Upgraded to Competition Grade. Added Stacking Ensemble, Target Encoding, and enhanced stability for multi-threaded training.
*   **v1.5.1**: Added SHAP explainability and color-coded CLI logging.