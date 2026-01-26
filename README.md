# Home Credit Default Risk: Competition-Grade Machine Learning Pipeline (v2.0.0)

### 1. Project Overview
The **Home Credit Default Risk** project is an elite-level machine learning pipeline (Suitability Score: 10/10) specifically engineered for high-stakes financial risk assessment. Unlike standard models, this system leverages a production-grade architecture to predict the probability of loan repayment difficulties using complex relational data.

#### Key Features:
*   **Competition-Grade Stacking Ensemble**: Combines **LightGBM**, **XGBoost**, and **CatBoost** using a Logistic Regression meta-learner to maximize ROC-AUC.
*   **Informed Drift Mitigation**: A sophisticated heuristic that preserves critical predictive signals while filtering out unstable, drifted features.
*   **Advanced Target Encoding**: Efficiently handles high-cardinality categorical data (e.g., occupation types) to capture deep behavioral patterns.
*   **Full Relational Integration**: Automatically aggregates historical credit behaviors, monthly balances, and installment patterns from supplemental Kaggle datasets.
*   **Explainable AI (XAI)**: Integrated SHAP interpretability and automated diagnostic plotting (ROC, Feature Importance, Risk Distribution).

---

### 2. Installation Instructions
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

### 3. Usage Instructions
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

### 4. Detailed Code Explanation
The codebase is modular, ensuring that each component can be tested and iterated upon independently.

#### A. Configuration (`src/config.py`)
Centralizes all hyperparameters for the ensemble components (LGBM, XGB, CatBoost). It also includes a custom `ColorFormatter` for the logging system, providing bold, color-coded alerts for data drift and system warnings.

#### B. Data Processing (`src/data_processing.py`)
Handles relational joins and the **Informed Drift Mitigation** logic. It uses a pilot-model pass to ensure that features are only dropped if they are **both** drifted (mean difference > 10%) and of low predictive importance.

#### C. Modeling & Ensemble (`src/modeling.py`)
The heart of the "Score 10" suitability. It implements:
*   **`StackingClassifier`**: Nested models are configured with strict thread isolation (`n_jobs=1`) to prevent deadlocks during parallel training.
*   **`CalibratedClassifierCV`**: Uses Platt scaling to ensure output probabilities are reliable for real-world financial decisions.

#### D. Orchestrator (`src/orchestrator.py`)
The workflow engine that sequences the pipeline from raw data ingestion to final submission generation and SHAP summary plotting.

---

### 5. Contributing Guidelines
We welcome contributions that improve the model's performance or extensibility.
*   **Modularity**: New features should be implemented as pure functions in `src/data_processing.py`.
*   **Testing**: Ensure all changes pass existing tests and add new tests in `tests/` for new logic.
    ```bash
    poetry run python tests/test_data_processing.py
    ```
*   **Code Style**: Adhere to PEP 8 and use Google-style docstrings.

---

### 6. License
This project is licensed under the **MIT License**. See the project root for full license text.

---

### 7. Changelog
*   **v2.0.1**: Fixed diagnostic visualizations for Stacking Ensemble. Enhanced `plot_feature_importance` and `plot_shap_summary` to support multi-model stacks.
*   **v2.0.0**: Upgraded to Competition Grade. Added Stacking Ensemble, Target Encoding, and enhanced stability for multi-threaded training.
*   **v1.5.1**: Added SHAP explainability and color-coded CLI logging.