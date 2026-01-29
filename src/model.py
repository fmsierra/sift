"""
Model Training Module - Phase 3
================================

Handles model training using HistGradientBoostingRegressor with MultiOutputRegressor.

Features:
    - Multi-output regression for predicting all columns simultaneously
    - Hyperparameter configuration via config file
    - Model persistence (save/load)
    - Training progress logging
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

logger = logging.getLogger(__name__)


class MultivariatePredictionModel:
    """
    Multivariate time series prediction model using HistGradientBoostingRegressor.
    
    Uses MultiOutputRegressor wrapper to predict all target columns simultaneously.
    """
    
    def __init__(
        self,
        max_iter: int = 100,
        max_depth: int = 10,
        learning_rate: float = 0.1,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.1,
        random_state: int = 42,
        early_stopping: bool = True,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 10
    ):
        """
        Initialize the model with hyperparameters.
        
        Args:
            max_iter: Maximum number of boosting iterations
            max_depth: Maximum depth of each tree
            learning_rate: Learning rate (shrinkage)
            min_samples_leaf: Minimum samples required in a leaf
            l2_regularization: L2 regularization strength
            random_state: Random seed for reproducibility
            early_stopping: Whether to use early stopping
            validation_fraction: Fraction of data for early stopping validation
            n_iter_no_change: Number of iterations without improvement before stopping
        """
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        
        self.model: Optional[MultiOutputRegressor] = None
        self.n_features_in_: Optional[int] = None
        self.n_targets_: Optional[int] = None
        self.training_info: Dict[str, Any] = {}
        self._is_fitted = False
    
    def _create_base_estimator(self) -> HistGradientBoostingRegressor:
        """Create the base HistGradientBoostingRegressor."""
        return HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            random_state=self.random_state,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            verbose=0
        )
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_jobs: int = -1
    ) -> 'MultivariatePredictionModel':
        """
        Train the model on the provided data.
        
        Args:
            X: Feature array of shape (n_samples, n_features)
            y: Target array of shape (n_samples, n_targets)
            n_jobs: Number of parallel jobs (-1 for all cores)
            
        Returns:
            Self for method chaining
        """
        start_time = datetime.now()
        
        logger.info("=" * 60)
        logger.info("STARTING MODEL TRAINING (Phase 3)")
        logger.info("=" * 60)
        logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
        logger.info(f"Hyperparameters:")
        logger.info(f"  - max_iter: {self.max_iter}")
        logger.info(f"  - max_depth: {self.max_depth}")
        logger.info(f"  - learning_rate: {self.learning_rate}")
        logger.info(f"  - min_samples_leaf: {self.min_samples_leaf}")
        logger.info(f"  - l2_regularization: {self.l2_regularization}")
        logger.info(f"  - early_stopping: {self.early_stopping}")
        
        self.n_features_in_ = X.shape[1]
        self.n_targets_ = y.shape[1] if len(y.shape) > 1 else 1
        
        # Create multi-output wrapper
        base_estimator = self._create_base_estimator()
        self.model = MultiOutputRegressor(base_estimator, n_jobs=n_jobs)
        
        logger.info(f"Training MultiOutputRegressor with {self.n_targets_} targets...")
        self.model.fit(X, y)
        
        end_time = datetime.now()
        training_duration = (end_time - start_time).total_seconds()
        
        # Store training info
        self.training_info = {
            'training_duration_seconds': training_duration,
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'n_targets': self.n_targets_,
            'trained_at': end_time.isoformat(),
            'hyperparameters': {
                'max_iter': self.max_iter,
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
                'min_samples_leaf': self.min_samples_leaf,
                'l2_regularization': self.l2_regularization
            }
        }
        
        # Check actual iterations for each estimator (if early stopping was used)
        if self.early_stopping:
            actual_iters = []
            for estimator in self.model.estimators_:
                actual_iters.append(estimator.n_iter_)
            self.training_info['actual_iterations'] = actual_iters
            logger.info(f"Actual iterations per target: {actual_iters}")
        
        self._is_fitted = True
        
        logger.info("=" * 60)
        logger.info(f"MODEL TRAINING COMPLETE in {training_duration:.2f} seconds")
        logger.info("=" * 60)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature array of shape (n_samples, n_features)
            
        Returns:
            Predictions array of shape (n_samples, n_targets)
        """
        if not self._is_fitted:
            raise ValueError("Model must be trained before prediction. Call fit() first.")
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, but got {X.shape[1]}"
            )
        
        return self.model.predict(X)
    
    def get_feature_importances(self) -> np.ndarray:
        """
        Get feature importances for each target.
        
        Returns:
            Array of shape (n_targets, n_features) containing importance scores
        """
        if not self._is_fitted:
            raise ValueError("Model must be trained first.")
        
        importances = []
        for estimator in self.model.estimators_:
            importances.append(estimator.feature_importances_)
        
        return np.array(importances)
    
    def save(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self._is_fitted:
            raise ValueError("Cannot save untrained model.")
        
        state = {
            'model': self.model,
            'hyperparameters': {
                'max_iter': self.max_iter,
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
                'min_samples_leaf': self.min_samples_leaf,
                'l2_regularization': self.l2_regularization,
                'random_state': self.random_state,
                'early_stopping': self.early_stopping,
                'validation_fraction': self.validation_fraction,
                'n_iter_no_change': self.n_iter_no_change
            },
            'n_features_in_': self.n_features_in_,
            'n_targets_': self.n_targets_,
            'training_info': self.training_info,
            '_is_fitted': self._is_fitted
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(state, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'MultivariatePredictionModel':
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded MultivariatePredictionModel instance
        """
        state = joblib.load(filepath)
        
        model = cls(**state['hyperparameters'])
        model.model = state['model']
        model.n_features_in_ = state['n_features_in_']
        model.n_targets_ = state['n_targets_']
        model.training_info = state['training_info']
        model._is_fitted = state['_is_fitted']
        
        logger.info(f"Model loaded from {filepath}")
        return model


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Dict[str, Any],
    save_path: Optional[str] = None
) -> MultivariatePredictionModel:
    """
    Train a model using configuration parameters.
    
    Args:
        X_train: Training features
        y_train: Training targets
        config: Model configuration dictionary
        save_path: Path to save the trained model (optional)
        
    Returns:
        Trained MultivariatePredictionModel
    """
    model_config = config.get('model', {})
    
    model = MultivariatePredictionModel(
        max_iter=model_config.get('max_iter', 100),
        max_depth=model_config.get('max_depth', 10),
        learning_rate=model_config.get('learning_rate', 0.1),
        min_samples_leaf=model_config.get('min_samples_leaf', 20),
        l2_regularization=model_config.get('l2_regularization', 0.1),
        random_state=model_config.get('random_state', 42),
        early_stopping=model_config.get('early_stopping', True),
        validation_fraction=model_config.get('validation_fraction', 0.1),
        n_iter_no_change=model_config.get('n_iter_no_change', 10)
    )
    
    model.fit(X_train, y_train)
    
    if save_path:
        model.save(save_path)
    
    return model


def print_model_summary(model: MultivariatePredictionModel) -> None:
    """
    Print a summary of the trained model.
    
    Args:
        model: Trained model instance
    """
    print("\n" + "=" * 50)
    print("MODEL SUMMARY")
    print("=" * 50)
    print(f"Model Type: MultiOutputRegressor(HistGradientBoostingRegressor)")
    print(f"Number of targets: {model.n_targets_}")
    print(f"Number of input features: {model.n_features_in_}")
    print(f"\nHyperparameters:")
    print(f"  - max_iter: {model.max_iter}")
    print(f"  - max_depth: {model.max_depth}")
    print(f"  - learning_rate: {model.learning_rate}")
    print(f"  - min_samples_leaf: {model.min_samples_leaf}")
    print(f"  - l2_regularization: {model.l2_regularization}")
    
    if model.training_info:
        print(f"\nTraining Info:")
        print(f"  - Duration: {model.training_info.get('training_duration_seconds', 'N/A'):.2f}s")
        print(f"  - Samples: {model.training_info.get('n_samples', 'N/A')}")
        if 'actual_iterations' in model.training_info:
            print(f"  - Actual iterations: {model.training_info['actual_iterations']}")
    
    print("=" * 50 + "\n")


if __name__ == "__main__":
    # Test model training with sample data
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create sample training data
    np.random.seed(42)
    n_samples = 1000
    n_features = 90  # 15 window * 6 columns
    n_targets = 6
    
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randn(n_samples, n_targets)
    
    print("Testing model training...")
    
    # Create config
    config = {
        'model': {
            'max_iter': 50,
            'max_depth': 5,
            'learning_rate': 0.1,
            'min_samples_leaf': 20,
            'l2_regularization': 0.1,
            'random_state': 42,
            'early_stopping': True
        }
    }
    
    model = train_model(X_train, y_train, config)
    print_model_summary(model)
    
    # Test prediction
    X_test = np.random.randn(10, n_features)
    predictions = model.predict(X_test)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample prediction: {predictions[0]}")
