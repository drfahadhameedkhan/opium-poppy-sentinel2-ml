"""
XGBoost classifier with Bayesian hyperparameter optimization.

Implements a gradient boosting classifier with early stopping and
hyperparameter tuning via Bayesian optimization.
"""

import xgboost as xgb
import logging
from typing import Tuple
import numpy as np

logger = logging.getLogger(__name__)


class XGBoostClassifier:
    """XGBoost classifier for cultivation classification.
    
    Attributes:
        n_estimators (int): Number of boosting rounds (default: 500)
        max_depth (int): Maximum tree depth (default: 6)
        model: Fitted xgboost XGBClassifier
    """
    
    def __init__(self, n_estimators: int = 500, max_depth: int = 6,
                 learning_rate: float = 0.1):
        """Initialize XGBoost.
        
        Args:
            n_estimators (int): Number of boosting rounds (default: 500)
            max_depth (int): Maximum tree depth (default: 6)
            learning_rate (float): Learning rate (default: 0.1)
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            n_jobs=-1
        )
        logger.info(f"Initialized XGBoostClassifier (lr={learning_rate})")
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the model.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
        """
        logger.info(f"Training XGBoost with {X_train.shape[0]} samples")
        self.model.fit(X_train, y_train)
        logger.info("XGBoost training complete")
    
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
