import pandas as pd
import numpy as np
import os
import sys
from sklearn.pipeline import Pipeline

# Add project root to path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.modeling import build_pipeline, try_build_model

def test_build_pipeline():
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5, 6],
        'B': ['x', 'y', 'z', 'x', 'y', 'z'],
        'C': [1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
    })
    cat_cols = ['B']
    num_cols = ['A', 'C']
    
    # In refactored code, build_pipeline returns a pipeline with a model
    pipeline = build_pipeline(cat_cols, num_cols)
    assert isinstance(pipeline, Pipeline)
    
    # We need a target for fit
    y = pd.Series([0, 1, 0, 1, 0, 1])
    pipeline.fit(df, y)
    
    # Check if we can predict
    proba = pipeline.predict_proba(df)
    assert proba.shape == (6, 2)

def test_try_build_model():
    # Test without calibration for simpler type checking if needed
    model_logreg = try_build_model(prefer_lightgbm=False, calibrate=False)
    from sklearn.linear_model import LogisticRegression
    assert isinstance(model_logreg, LogisticRegression)
    
    # Test with calibration
    model_calibrated = try_build_model(prefer_lightgbm=False, calibrate=True)
    from sklearn.calibration import CalibratedClassifierCV
    assert isinstance(model_calibrated, CalibratedClassifierCV)

if __name__ == "__main__":
    test_build_pipeline()
    print("test_build_pipeline passed!")
    test_try_build_model()
    print("test_try_build_model passed!")
