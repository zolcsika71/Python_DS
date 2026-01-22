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
        'A': [1, 2, 3],
        'B': ['x', 'y', 'z'],
        'C': [1.1, 2.2, 3.3]
    })
    cat_cols = ['B']
    num_cols = ['A', 'C']
    
    # In refactored code, build_pipeline returns a pipeline with a model
    pipeline = build_pipeline(cat_cols, num_cols)
    assert isinstance(pipeline, Pipeline)
    
    # We need a target for fit
    y = pd.Series([0, 1, 0])
    pipeline.fit(df, y)
    
    # Check if we can predict
    proba = pipeline.predict_proba(df)
    assert proba.shape == (3, 2)

def test_try_build_model():
    model_logreg = try_build_model(prefer_lightgbm=False)
    from sklearn.linear_model import LogisticRegression
    assert isinstance(model_logreg, LogisticRegression)
    
    # LightGBM might not be installed in all environments, so we just check it doesn't crash
    model_lgbm = try_build_model(prefer_lightgbm=True)
    assert model_lgbm is not None

if __name__ == "__main__":
    test_build_pipeline()
    print("test_build_pipeline passed!")
    test_try_build_model()
    print("test_try_build_model passed!")
