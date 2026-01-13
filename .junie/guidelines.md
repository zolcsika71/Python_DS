### Project Guidelines

#### Build and Configuration
This project uses **Poetry** for dependency management and packaging.

1.  **Environment Setup**:
    - Ensure you have Poetry installed.
    - Install dependencies:
      ```bash
      poetry install
      ```
    - The project requires Python >= 3.13.

2.  **Data Requirements**:
    - The project expects Kaggle's "Home Credit Default Risk" dataset.
    - Place `application_train.csv` and `application_test.csv` in the `data/` directory (or specify a custom path using `--data-dir`).

3.  **Running the Script**:
    - Basic execution:
      ```bash
      poetry run python main.py
      ```
    - To use LightGBM (recommended if available):
      ```bash
      poetry run python main.py --prefer-lightgbm
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

2.  **Adding New Tests**:
    - Create a new file prefixed with `test_` (e.g., `test_feature_engineering.py`).
    - Use standard `pytest` assertions.
    - If a test requires specific scikit-learn functionality, ensure it's compatible with the installed version (currently using `1.7.2` to avoid internal import issues in some environments).

3.  **Example Test**:
    The following test verifies the `fix_known_anomalies` function in `main.py`:
    ```python
    import pandas as pd
    import numpy as np
    from main import fix_known_anomalies

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
- **Code Style**: Follow standard PEP 8 guidelines. The project uses a functional approach for data processing steps.
- **Model Pipeline**: The `build_pipeline` function dynamically creates a `ColumnTransformer` based on the input dataframe's types (categorical vs. numeric).
- **Anomalies**: Always use `fix_known_anomalies` after loading the data to handle specific placeholders like `365243` in `DAYS_EMPLOYED`.
- **Output**: Submissions are saved in the `submissions/` directory with a timestamped filename by default.
- **Plots**: Training results, including ROC curves and feature importance, are saved in the `plots/` directory.
