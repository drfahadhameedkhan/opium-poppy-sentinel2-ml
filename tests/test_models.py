"""
Tests for machine learning models.
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification
from src.models import RandomForestClassifier, XGBoostClassifier


class TestRandomForest:
    """Test Random Forest classifier."""
    
    def setup_method(self):
        """Setup test data."""
        X, y = make_classification(n_samples=100, n_features=10, n_classes=2,
                                   n_informative=5, random_state=42)
        self.X_train = X[:80]
        self.y_train = y[:80]
        self.X_test = X[80:]
        self.y_test = y[80:]
    
    def test_rf_initialization(self):
        """Test RF model initialization."""
        clf = RandomForestClassifier(n_estimators=10, max_depth=5)
        assert clf.n_estimators == 10
        assert clf.max_depth == 5
    
    def test_rf_training(self):
        """Test RF model training."""
        clf = RandomForestClassifier(n_estimators=10)
        clf.fit(self.X_train, self.y_train)
        predictions = clf.predict(self.X_test)
        assert predictions.shape == self.y_test.shape
    
    def test_rf_predictions_valid(self):
        """Test RF predictions are valid."""
        clf = RandomForestClassifier(n_estimators=10)
        clf.fit(self.X_train, self.y_train)
        predictions = clf.predict(self.X_test)
        assert np.all(np.isin(predictions, np.unique(self.y_train)))
    
    def test_rf_predict_proba(self):
        """Test RF probability predictions."""
        clf = RandomForestClassifier(n_estimators=10)
        clf.fit(self.X_train, self.y_train)
        proba = clf.predict_proba(self.X_test)
        assert proba.shape[0] == self.X_test.shape[0]
        assert np.all((proba >= 0) & (proba <= 1))


class TestXGBoost:
    """Test XGBoost classifier."""
    
    def setup_method(self):
        """Setup test data."""
        X, y = make_classification(n_samples=100, n_features=10, n_classes=2,
                                   n_informative=5, random_state=42)
        self.X_train = X[:80]
        self.y_train = y[:80]
        self.X_test = X[80:]
    
    def test_xgb_initialization(self):
        """Test XGBoost initialization."""
        clf = XGBoostClassifier(n_estimators=10, learning_rate=0.05)
        assert clf.n_estimators == 10
        assert clf.learning_rate == 0.05
    
    def test_xgb_training(self):
        """Test XGBoost model training."""
        clf = XGBoostClassifier(n_estimators=10)
        clf.fit(self.X_train, self.y_train)
        predictions = clf.predict(self.X_test)
        assert predictions.shape[0] == self.X_test.shape[0]
