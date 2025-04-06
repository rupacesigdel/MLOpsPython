import pytest
import pandas as pd
import numpy as np
from copy import deepcopy
from src.train import CONFIG

@pytest.fixture
def sample_data():
    """Generate test data with correct column names"""
    return pd.DataFrame({
        'feature1': np.random.normal(0, 1, 200),
        'feature2': np.random.uniform(0, 1, 200),
        'target': np.random.randint(0, 2, 200) 
    })

@pytest.fixture
def original_config():
    """Preserve original configuration"""
    return deepcopy(CONFIG)

@pytest.fixture(autouse=True)
def restore_config(original_config):
    """Auto-restore config after each test"""
    yield
    CONFIG.clear()
    CONFIG.update(original_config)


