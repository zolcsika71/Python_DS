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
