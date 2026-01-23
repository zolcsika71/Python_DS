### Project Guidelines

#### Build and Configuration
This project uses **Poetry** for dependency management and packaging.

1.  **Environment Setup**:
    - Ensure you have Poetry installed.
    - Install dependencies:
      ```bash
      poetry install
      ```
    - The project requires Python >= 3.12.

2.  **Data Requirements**:
    - The project expects Kaggle's "Home Credit Default Risk" dataset.
    - Place the following files in the `data/` directory:
        - `application_train.csv` (Required)
        - `application_test.csv` (Required)
        - `bureau.csv` (Optional)
        - `bureau_balance.csv` (Optional)
        - `previous_application.csv` (Optional)
        - `POS_CASH_balance.csv` (Optional)
        - `installments_payments.csv` (Optional)
        - `credit_card_balance.csv` (Optional)

3.  **Running the Script**:
    - Basic execution:
      ```bash
      poetry run python main.py
      ```
    - To use LightGBM (recommended if available):
      ```bash
      poetry run python main.py --prefer-lightgbm
      ```
    - Advanced execution with custom folds and output:
      ```bash
      poetry run python main.py --prefer-lightgbm --folds 5 --out submissions/my_sub.csv
      ```
    - Check all options:
      ```bash
      poetry run python main.py --help
      ```

#### Testing Information
The project uses `pytest` for automated testing.

1.  **Running Tests**:
    - Execute all tests:
      ```bash
      poetry run pytest
      ```
    - Run individual test files:
      ```bash
      poetry run python tests/test_data_processing.py
      poetry run python tests/test_modeling.py
      poetry run python tests/test_config.py
      poetry run python tests/test_visualization.py
      poetry run python tests/test_orchestrator.py
      ```

2.  **Adding New Tests**:
    - Create a new file in the `tests/` directory prefixed with `test_`.
    - Use standard `pytest` assertions.
    - If a test requires specific scikit-learn functionality, ensure it's compatible with the installed version.

3.  **Example Test**:
    The following test verifies the `fix_known_anomalies` function in `src/data_processing.py`:
    ```python
    import pandas as pd
    import numpy as np
    from src.data_processing import fix_known_anomalies

    def test_fix_known_anomalies():
        # DAYS_EMPLOYED 365243 is a known anomaly placeholder in this dataset
        df = pd.DataFrame({
            'DAYS_EMPLOYED': [100, 365243, -500],
        })
        
        fixed_df = fix_known_anomalies(df)
        
        # Check if 365243 is replaced with NaN
        assert np.isnan(fixed_df.loc[1, 'DAYS_EMPLOYED'])
        # Check if the flag column is created correctly
        assert fixed_df.loc[1, 'DAYS_EMPLOYED_ANOM'] == 1
        assert fixed_df.loc[0, 'DAYS_EMPLOYED_ANOM'] == 0
    ```

#### Additional Development Information
- **Modular Architecture**: The project is organized into modules under `src/`:
    - `config.py`: Centralized configuration, logging setup, and hyperparameters.
    - `data_processing.py`: Data loading, cleaning, feature engineering, and drift detection.
    - `modeling.py`: Pipeline building and cross-validation.
    - `visualization.py`: Plotting functions (ROC, Importance, SHAP, etc.).
    - `orchestrator.py`: High-level workflow orchestration and top-10 analysis.
    - `top_10_analysis.py`: Standalone script for risk analysis.
    - `process_target.py`: Utility for post-processing submissions.
- **Automated Drift Mitigation**: The pipeline automatically detects data drift and drops low-importance drifted features.
- **SHAP Explainability**: Global feature importance is visualized using SHAP summary plots.
- **Probability Calibration**: Uses Platt scaling for well-calibrated risk estimates.
- **Code Style**: Follow standard PEP 8 guidelines. The project uses a functional approach for data processing steps.
- **Logging**: Use the centralized `logger` from `src.config`. `INFO` level messages are color-coded green in the console.
- **Model Pipeline**: The `build_pipeline` function in `src/modeling.py` dynamically creates a `ColumnTransformer` based on the input dataframe's types.
- **Anomalies**: Always use `fix_known_anomalies` (from `src.data_processing`) after loading the data to handle specific placeholders.
- **Output**: Submissions are saved in the `submissions/` directory.
- **Plots**: Training results and analysis visualizations are saved in the `plots/` directory.
