import pandas as pd
import numpy as np
from src.modeling import build_pipeline
from src.config import CONFIG
import time

def reproduce():
    print("Generating synthetic data...")
    X = pd.DataFrame(np.random.rand(1000, 20), columns=[f'feat_{i}' for i in range(20)])
    y = np.random.randint(0, 2, 1000)
    
    cat_cols = []
    num_cols = X.columns.tolist()
    
    # Use fewer estimators for faster reproduction
    CONFIG.lgbm_params['n_estimators'] = 10
    CONFIG.xgb_params['n_estimators'] = 10
    CONFIG.cat_params['iterations'] = 10
    
    print("Building ensemble pipeline...")
    # use_ensemble=True triggers the StackingClassifier
    clf = build_pipeline(cat_cols, num_cols, use_ensemble=True, calibrate=True)
    
    print("Starting fit (this is where it might hang)...")
    start_time = time.time()
    try:
        clf.fit(X, y)
        duration = time.time() - start_time
        print(f"Fit completed successfully in {duration:.2f} seconds.")
    except KeyboardInterrupt:
        print("\nFit interrupted by user (KeyboardInterrupt).")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    reproduce()
