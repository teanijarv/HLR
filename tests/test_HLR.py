import pytest
from HLR.regression import HierarchicalLinearRegression
import pandas as pd

# Example test data
df = pd.DataFrame({
    'PTS': [120, 115, 130, 112, 104],
    'ORB': [10, 11, 8, 10, 9],
    'BLK': [5, 3, 6, 7, 4],
    'W': [50, 45, 55, 47, 44]
})
X = {1: ['PTS'], 2: ['PTS', 'ORB'], 3: ['PTS', 'ORB', 'BLK']}
y = 'W'

def test_model_initialization():
    """Test initialization of HierarchicalLinearRegression model."""
    hlr = HierarchicalLinearRegression(df, X, y)
    assert hlr.data is not None
    assert hlr.outcome_var == y

def test_fit_models():
    """Test fitting models."""
    hlr = HierarchicalLinearRegression(df, X, y)
    model_results = hlr.fit_models()
    assert len(model_results) == len(X)