from src.train import load_and_validate_data

# tests/test_train.py
def test_data_loading(sample_data, tmp_path, monkeypatch):
    # Save test data
    test_path = tmp_path / "test_data.csv"
    sample_data.to_csv(test_path, index=False)
    
    # Temporarily patch the configuration
    from src.train import CONFIG, DATA_PATH
    CONFIG['data']['target_col'] = 'target'  # Must match your fixture column
    CONFIG['data']['min_samples'] = 5  # Lower threshold for tests
    
    # Use monkeypatch to safely modify DATA_PATH
    monkeypatch.setattr('src.train.DATA_PATH', str(test_path))
    
    # Import AFTER patching
    from src.train import load_and_validate_data
    X_train, X_test, y_train, y_test = load_and_validate_data()
    
    assert len(X_train) > 0
    assert len(y_test) > 0