import pandas as pd
import numpy as np
import os
import sys

# Add project root to path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_processing import fix_known_anomalies, select_features_by_drift

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

def test_select_features_by_drift():
    # Setup data with drift
    train = pd.DataFrame({'feat1': [1, 2, 3], 'feat2': [10, 20, 30]})
    test = pd.DataFrame({'feat1': [1, 2, 3], 'feat2': [100, 200, 300]}) # High drift in feat2
    
    # Importance data (feat2 is low importance)
    importances = pd.DataFrame({
        'feature': ['feat1', 'feat2'],
        'importance': [100, 5]
    })
    
    # Run selection
    train_fixed, test_fixed, dropped = select_features_by_drift(
        train, test, drift_threshold=0.1, importance_threshold=10.0, importances=importances
    )
    
    assert 'feat2' in dropped
    assert 'feat2' not in train_fixed.columns
    assert 'feat1' in train_fixed.columns
    
    # Test missing feature in importance
    importances_missing = pd.DataFrame({
        'feature': ['feat1'],
        'importance': [100]
    })
    # feat2 is drifted but missing in importance -> should be treated as NaN importance and dropped
    train_fixed, test_fixed, dropped = select_features_by_drift(
        train, test, drift_threshold=0.1, importance_threshold=10.0, importances=importances_missing
    )
    assert 'feat2' in dropped

if __name__ == "__main__":
    test_fix_known_anomalies()
    print("test_fix_known_anomalies passed!")
    test_select_features_by_drift()
    print("test_select_features_by_drift passed!")