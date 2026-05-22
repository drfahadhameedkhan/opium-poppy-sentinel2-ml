"""
Random Forest classifier with hyperparameter optimization.

Implements an ensemble Random Forest classifier with feature importance
analysis and hyperparameter tuning capabilities.
"""

from sklearn.ensemble import RandomForestClassifier as RF
import logging
from typing import Tuple
import numpy as np

logger = logging.getLogger(__name__)


class RandomForestClassifier:
    """Random Forest classifier for cultivation classification.
    
    Attributes:
        n_estimators (int): Number of trees in the forest (default: 500)
        max_depth (int): Maximum depth of trees (default: 20)
        model: Fitted sklearn RandomForestClassifier
    """
    
    def __init__(self, n_estimators: int = 500, max_depth: int = 20):
        """Initialize Random Forest.
        
        Args:
            n_estimators (int): Number of trees (default: 500)
            max_depth (int): Maximum tree depth (default: 20)
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = RF(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        logger.info(f"Initialized RandomForestClassifier ({n_estimators} trees)")
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the model.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
        """
        logger.info(f"Training RF with {X_train.shape[0]} samples")
        self.model.fit(X_train, y_train)
        logger.info("RF training complete")
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X_test (np.ndarray): Test features
            
        Returns:
            np.ndarray: Predicted class labels
        """
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X_test (np.ndarray): Test features
            
        Returns:
            np.ndarray: Class probabilities
        """
        return self.model.predict_proba(X_test)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores.
        
        Returns:
            np.ndarray: Feature importance values
        """
        return self.model.feature_importances_
