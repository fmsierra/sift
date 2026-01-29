"""
Test Suite for Preprocessing Module
=====================================

Tests for the TimeSeriesPreprocessor class and preprocessing functions.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import TimeSeriesPreprocessor, preprocess_pipeline


class TestTimeSeriesPreprocessor:
    """Tests for TimeSeriesPreprocessor class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 100
        return pd.DataFrame({
            'col_1': np.random.randn(n_samples) + 10,
            'col_2': np.random.randn(n_samples) + 20,
            'col_3': np.random.randn(n_samples) + 30,
            'col_4': np.random.randn(n_samples) + 40,
            'col_5': np.random.randn(n_samples) + 50,
            'col_6': np.random.randn(n_samples) + 60
        })
    
    @pytest.fixture
    def preprocessor(self):
        """Create a preprocessor instance."""
        return TimeSeriesPreprocessor(
            window_size=15,
            normalize=True,
            train_split=0.9
        )
    
    def test_init(self, preprocessor):
        """Test preprocessor initialization."""
        assert preprocessor.window_size == 15
        assert preprocessor.normalize == True
        assert preprocessor.train_split == 0.9
        assert preprocessor._is_fitted == False
    
    def test_fit(self, preprocessor, sample_data):
        """Test fitting the preprocessor."""
        preprocessor.fit(sample_data)
        
        assert preprocessor._is_fitted == True
        assert preprocessor.n_features == 6
        assert len(preprocessor.feature_columns) == 6
        assert preprocessor.scaler is not None
    
    def test_transform_before_fit(self, preprocessor, sample_data):
        """Test that transform raises error before fit."""
        with pytest.raises(ValueError, match="must be fitted"):
            preprocessor.transform(sample_data)
    
    def test_fit_transform(self, preprocessor, sample_data):
        """Test fit_transform returns normalized data."""
        data = preprocessor.fit_transform(sample_data)
        
        assert data.shape == (100, 6)
        assert data.min() >= 0
        assert data.max() <= 1
    
    def test_inverse_transform(self, preprocessor, sample_data):
        """Test inverse transform recovers original scale."""
        data_normalized = preprocessor.fit_transform(sample_data)
        data_recovered = preprocessor.inverse_transform(data_normalized)
        
        np.testing.assert_array_almost_equal(
            data_recovered,
            sample_data.values,
            decimal=10
        )
    
    def test_sliding_window_shape(self, preprocessor, sample_data):
        """Test sliding window creates correct shapes."""
        data = preprocessor.fit_transform(sample_data)
        X, y = preprocessor.create_sliding_window(data)
        
        # Expected: n_samples - window_size
        expected_samples = 100 - 15
        expected_features = 15 * 6  # window_size * n_features
        expected_targets = 6
        
        assert X.shape == (expected_samples, expected_features)
        assert y.shape == (expected_samples, expected_targets)
    
    def test_train_test_split(self, preprocessor, sample_data):
        """Test chronological train/test split."""
        data = preprocessor.fit_transform(sample_data)
        X, y = preprocessor.create_sliding_window(data)
        X_train, X_test, y_train, y_test = preprocessor.prepare_train_test_split(X, y)
        
        total = len(X)
        train_expected = int(total * 0.9)
        test_expected = total - train_expected
        
        assert len(X_train) == train_expected
        assert len(X_test) == test_expected
        assert len(y_train) == train_expected
        assert len(y_test) == test_expected
    
    def test_final_window(self, preprocessor, sample_data):
        """Test final window preparation."""
        data = preprocessor.fit_transform(sample_data)
        final_window = preprocessor.prepare_final_window(data)
        
        expected_features = 15 * 6
        assert final_window.shape == (1, expected_features)
    
    def test_feature_names(self, preprocessor, sample_data):
        """Test feature name generation."""
        preprocessor.fit(sample_data)
        names = preprocessor.get_feature_names()
        
        assert len(names) == 15 * 6
        assert 'col_1_lag_15' in names
        assert 'col_6_lag_1' in names
    
    def test_save_load(self, preprocessor, sample_data):
        """Test saving and loading preprocessor."""
        preprocessor.fit_transform(sample_data)
        
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            temp_path = f.name
        
        try:
            preprocessor.save(temp_path)
            loaded = TimeSeriesPreprocessor.load(temp_path)
            
            assert loaded.window_size == preprocessor.window_size
            assert loaded.normalize == preprocessor.normalize
            assert loaded._is_fitted == True
        finally:
            os.unlink(temp_path)


class TestPreprocessPipeline:
    """Tests for the preprocess_pipeline function."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        return pd.DataFrame({
            'col_1': np.random.randn(200) + 10,
            'col_2': np.random.randn(200) + 20,
            'col_3': np.random.randn(200) + 30,
            'col_4': np.random.randn(200) + 40,
            'col_5': np.random.randn(200) + 50,
            'col_6': np.random.randn(200) + 60
        })
    
    def test_pipeline_returns_expected_keys(self, sample_data):
        """Test that pipeline returns all expected keys."""
        result = preprocess_pipeline(sample_data, window_size=10, train_split=0.8)
        
        expected_keys = [
            'X_train', 'X_test', 'y_train', 'y_test',
            'preprocessor', 'feature_names', 'final_window', 'data_normalized'
        ]
        
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
    
    def test_pipeline_shapes(self, sample_data):
        """Test that pipeline outputs have correct shapes."""
        result = preprocess_pipeline(
            sample_data,
            window_size=10,
            normalize=True,
            train_split=0.8
        )
        
        # Total samples after windowing: 200 - 10 = 190
        # Train: 80% of 190 = 152
        # Test: 20% of 190 = 38
        
        assert result['X_train'].shape[0] == 152
        assert result['X_test'].shape[0] == 38
        assert result['X_train'].shape[1] == 10 * 6  # window * features


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
