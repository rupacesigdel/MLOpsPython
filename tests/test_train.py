# tests/test_train.py
import pytest
from train import load_and_validate_data

def test_data_loading():
    X_train, X_test, y_train, y_test = load_and_validate_data()
    assert len(X_train) > 0
    assert len(y_test) > 0