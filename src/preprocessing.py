"""
Data Preprocessing Module - Phase 2
====================================

Handles data transformation, feature engineering, and train/test splitting.

Functions:
    - normalize_data: MinMax scaling to 0-1 range
    - create_sliding_window: Generate lag features for time series
    - prepare_train_test_split: Chronological data splitting
    - inverse_transform: Convert predictions back to original scale
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

logger = logging.getLogger(__name__)


class TimeSeriesPreprocessor:
    """
    Preprocessing pipeline for multivariate time series data.
    
    Handles normalization, windowing, and train/test splitting
    while maintaining proper temporal order.
    """
    
    def __init__(
        self,
        window_size: int = 15,
        normalize: bool = True,
        train_split: float = 0.9
    ):
        """
        Initialize the preprocessor.
        
        Args:
            window_size: Number of past time steps to use as features
            normalize: Whether to apply MinMax normalization
            train_split: Fraction of data for training (rest for testing)
        """
        self.window_size = window_size
        self.normalize = normalize
        self.train_split = train_split
        
        self.scaler: Optional[MinMaxScaler] = None
        self.feature_columns: Optional[List[str]] = None
        self.n_features: Optional[int] = None
        self._is_fitted = False
    
    def fit(self, df: pd.DataFrame) -> 'TimeSeriesPreprocessor':
        """
        Fit the preprocessor to the data (learn scaling parameters).
        
        Args:
            df: DataFrame with time series data
            
        Returns:
            Self for method chaining
        """
        self.feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.n_features = len(self.feature_columns)
        
        if self.normalize:
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            self.scaler.fit(df[self.feature_columns].values)
            logger.info("Fitted MinMaxScaler to data")
        
        self._is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted parameters.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Normalized numpy array
        """
        if not self._is_fitted:
            raise ValueError("Preprocessor must be fitted before transform. Call fit() first.")
        
        data = df[self.feature_columns].values
        
        if self.normalize and self.scaler is not None:
            data = self.scaler.transform(data)
        
        return data
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            df: DataFrame to fit and transform
            
        Returns:
            Normalized numpy array
        """
        self.fit(df)
        return self.transform(df)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Convert normalized data back to original scale.
        
        Args:
            data: Normalized data array
            
        Returns:
            Data in original scale
        """
        if not self._is_fitted:
            raise ValueError("Preprocessor must be fitted before inverse_transform.")
        
        if self.normalize and self.scaler is not None:
            return self.scaler.inverse_transform(data)
        return data
    
    def create_sliding_window(
        self,
        data: np.ndarray,
        target_offset: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding window features and targets for supervised learning.
        
        For each position, uses the past `window_size` values as features
        to predict the next values.
        
        Args:
            data: 2D array of shape (n_samples, n_features)
            target_offset: How many steps ahead to predict (default=1)
            
        Returns:
            Tuple of (X, y) where:
                X: Features array of shape (n_samples - window_size, window_size * n_features)
                y: Targets array of shape (n_samples - window_size, n_features)
        """
        n_samples, n_features = data.shape
        
        if n_samples <= self.window_size:
            raise ValueError(
                f"Data length ({n_samples}) must be greater than window size ({self.window_size})"
            )
        
        X_list = []
        y_list = []
        
        for i in range(self.window_size, n_samples):
            # Features: flatten the window of all features
            window = data[i - self.window_size:i, :]  # Shape: (window_size, n_features)
            X_list.append(window.flatten())  # Shape: (window_size * n_features,)
            
            # Target: all features at the next time step
            y_list.append(data[i, :])  # Shape: (n_features,)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        logger.info(
            f"Created sliding window dataset: X shape {X.shape}, y shape {y.shape}"
        )
        
        return X, y
    
    def prepare_train_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data chronologically into train and test sets.
        
        IMPORTANT: Never shuffle time series data!
        
        Args:
            X: Feature array
            y: Target array
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        n_samples = len(X)
        split_idx = int(n_samples * self.train_split)
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        logger.info(
            f"Train/Test split: {len(X_train)} train samples, {len(X_test)} test samples"
        )
        
        return X_train, X_test, y_train, y_test
    
    def prepare_final_window(self, data: np.ndarray) -> np.ndarray:
        """
        Prepare the final window for making predictions.
        
        Uses the last `window_size` rows to create input for predicting
        the next time step.
        
        Args:
            data: Full normalized data array
            
        Returns:
            Single feature vector for prediction
        """
        if len(data) < self.window_size:
            raise ValueError(
                f"Data length ({len(data)}) must be >= window size ({self.window_size})"
            )
        
        # Get last window_size rows and flatten
        final_window = data[-self.window_size:, :]
        return final_window.flatten().reshape(1, -1)
    
    def get_feature_names(self) -> List[str]:
        """
        Generate feature names for the windowed dataset.
        
        Returns:
            List of feature names in format 'column_lag_N'
        """
        if self.feature_columns is None:
            raise ValueError("Preprocessor must be fitted first.")
        
        feature_names = []
        for lag in range(self.window_size, 0, -1):
            for col in self.feature_columns:
                feature_names.append(f"{col}_lag_{lag}")
        
        return feature_names
    
    def save(self, filepath: str) -> None:
        """
        Save the preprocessor state to disk.
        
        Args:
            filepath: Path to save the preprocessor
        """
        state = {
            'window_size': self.window_size,
            'normalize': self.normalize,
            'train_split': self.train_split,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'n_features': self.n_features,
            '_is_fitted': self._is_fitted
        }
        joblib.dump(state, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'TimeSeriesPreprocessor':
        """
        Load a preprocessor from disk.
        
        Args:
            filepath: Path to the saved preprocessor
            
        Returns:
            Loaded TimeSeriesPreprocessor instance
        """
        state = joblib.load(filepath)
        
        preprocessor = cls(
            window_size=state['window_size'],
            normalize=state['normalize'],
            train_split=state['train_split']
        )
        preprocessor.scaler = state['scaler']
        preprocessor.feature_columns = state['feature_columns']
        preprocessor.n_features = state['n_features']
        preprocessor._is_fitted = state['_is_fitted']
        
        logger.info(f"Preprocessor loaded from {filepath}")
        return preprocessor


def preprocess_pipeline(
    df: pd.DataFrame,
    window_size: int = 15,
    normalize: bool = True,
    train_split: float = 0.9,
    save_preprocessor: Optional[str] = None
) -> Dict[str, Any]:
    """
    Complete preprocessing pipeline for time series data.
    
    Args:
        df: Raw DataFrame
        window_size: Number of lag features
        normalize: Whether to normalize data
        train_split: Train/test split ratio
        save_preprocessor: Path to save the fitted preprocessor
        
    Returns:
        Dictionary containing:
            - X_train, X_test, y_train, y_test: Split datasets
            - preprocessor: Fitted TimeSeriesPreprocessor
            - feature_names: Names of windowed features
            - final_window: Last window for prediction
    """
    logger.info("=" * 60)
    logger.info("STARTING DATA PREPROCESSING (Phase 2)")
    logger.info("=" * 60)
    
    # Initialize preprocessor
    preprocessor = TimeSeriesPreprocessor(
        window_size=window_size,
        normalize=normalize,
        train_split=train_split
    )
    
    # Fit and transform
    logger.info(f"Normalizing data: {normalize}")
    logger.info(f"Window size: {window_size}")
    data_normalized = preprocessor.fit_transform(df)
    
    # Create sliding window
    X, y = preprocessor.create_sliding_window(data_normalized)
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.prepare_train_test_split(X, y)
    
    # Prepare final window for prediction
    final_window = preprocessor.prepare_final_window(data_normalized)
    
    # Save preprocessor if path provided
    if save_preprocessor:
        preprocessor.save(save_preprocessor)
    
    result = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'preprocessor': preprocessor,
        'feature_names': preprocessor.get_feature_names(),
        'final_window': final_window,
        'data_normalized': data_normalized
    }
    
    logger.info("=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info(f"  Training samples: {len(X_train)}")
    logger.info(f"  Test samples: {len(X_test)}")
    logger.info(f"  Features per sample: {X_train.shape[1]}")
    logger.info(f"  Targets per sample: {y_train.shape[1]}")
    logger.info("=" * 60)
    
    return result


def print_preprocessing_summary(result: Dict[str, Any]) -> None:
    """
    Print a summary of the preprocessing results.
    
    Args:
        result: Dictionary from preprocess_pipeline
    """
    print("\n" + "=" * 50)
    print("PREPROCESSING SUMMARY")
    print("=" * 50)
    print(f"Training samples: {result['X_train'].shape[0]}")
    print(f"Test samples: {result['X_test'].shape[0]}")
    print(f"Features per sample: {result['X_train'].shape[1]}")
    print(f"Targets per sample: {result['y_train'].shape[1]}")
    print(f"\nWindow size: {result['preprocessor'].window_size}")
    print(f"Normalization: {result['preprocessor'].normalize}")
    print(f"Train/Test split: {result['preprocessor'].train_split}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    # Test preprocessing with sample data
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create sample data
    np.random.seed(42)
    n_samples = 500
    
    sample_df = pd.DataFrame({
        'col_1': np.cumsum(np.random.randn(n_samples)) + 10,
        'col_2': np.sin(np.linspace(0, 10*np.pi, n_samples)) + np.random.randn(n_samples) * 0.1 + 20,
        'col_3': np.random.randn(n_samples) + 30,
        'col_4': np.cumsum(np.random.randn(n_samples)) * 0.5 + 40,
        'col_5': np.cos(np.linspace(0, 8*np.pi, n_samples)) + np.random.randn(n_samples) * 0.2 + 50,
        'col_6': np.random.randn(n_samples) * 2 + 60
    })
    
    print("Testing preprocessing pipeline...")
    result = preprocess_pipeline(
        sample_df,
        window_size=15,
        normalize=True,
        train_split=0.9
    )
    
    print_preprocessing_summary(result)
    
    # Test inverse transform
    print("Testing inverse transform...")
    sample_pred = result['y_test'][0:3]
    original_scale = result['preprocessor'].inverse_transform(sample_pred)
    print(f"Sample prediction (normalized): {sample_pred[0][:3]}...")
    print(f"Sample prediction (original scale): {original_scale[0][:3]}...")
