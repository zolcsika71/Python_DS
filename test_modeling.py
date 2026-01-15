import pandas as pd
import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from src.modeling import build_pipeline, try_build_model

def test_build_pipeline():
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': ['x', 'y', 'z'],
        'C': [1.1, 2.2, 3.3]
    })
    cat_cols = ['B']
    num_cols = ['A', 'C']
    
    pipeline = build_pipeline(cat_cols, num_cols)
    assert isinstance(pipeline, Pipeline)
    
    transformed = pipeline.fit_transform(df)
    assert 'B_x' in transformed.columns or 'B' in transformed.columns # Depends on onehot output
    assert 'A' in transformed.columns
    assert 'C' in transformed.columns

def test_try_build_model():
    model_logreg = try_build_model(prefer_lightgbm=False)
    from sklearn.linear_model import LogisticRegression
    assert isinstance(model_logreg, LogisticRegression)
    
    # LightGBM might not be installed in all environments, so we just check it doesn't crash
    model_lgbm = try_build_model(prefer_lightgbm=True)
    assert model_lgbm is not None
