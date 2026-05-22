"""
Tests for model evaluation module.
"""

import pytest
import numpy as np
from src.evaluation import AccuracyAssessor, SpatialBlockCV


class TestAccuracyAssessor:
    """Test accuracy assessment functionality."""
    
    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)
        self.y_true = np.array([0, 1, 1, 2, 2, 2, 1, 0, 1, 2])
        self.y_pred = np.array([0, 1, 1, 2, 1, 2, 1, 0, 2, 2])
    
    def test_assessor_initialization(self):
        """Test AccuracyAssessor initialization."""
        assessor = AccuracyAssessor(self.y_true, self.y_pred)
        assert len(assessor.y_true) == len(self.y_true)
        assert assessor.area_weights == True
    
    def test_overall_accuracy(self):
        """Test overall accuracy computation."""
        assessor = AccuracyAssessor(self.y_true, self.y_pred)
        oa = assessor.overall_accuracy()
        assert 0 <= oa <= 1
    
    def test_kappa_coefficient(self):
        """Test Kappa coefficient computation."""
        assessor = AccuracyAssessor(self.y_true, self.y_pred)
        kappa = assessor.kappa_coefficient()
        assert -1 <= kappa <= 1
    
    def test_confusion_matrix(self):
        """Test confusion matrix generation."""
        assessor = AccuracyAssessor(self.y_true, self.y_pred)
        cm = assessor.confusion_matrix()
        assert cm.shape[0] == cm.shape[1]  # Square matrix
    
    def test_compute_all_metrics(self):
        """Test computation of all metrics."""
        assessor = AccuracyAssessor(self.y_true, self.y_pred)
        metrics = assessor.compute_all_metrics()
        assert 'overall_accuracy' in metrics
        assert 'kappa' in metrics


class TestSpatialBlockCV:
    """Test spatial block cross-validation."""
    
    def test_initialization(self):
        """Test SpatialBlockCV initialization."""
        cv = SpatialBlockCV(block_size=50, n_splits=5)
        assert cv.block_size == 50
        assert cv.n_splits == 5
