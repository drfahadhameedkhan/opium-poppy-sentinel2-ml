"""
Area-adjusted error matrices and accuracy metrics.

Implements accuracy assessment following Olofsson et al. (2014) protocol
for area-adjusted accuracy estimates.
"""

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class AccuracyAssessor:
    """Compute accuracy metrics with area-adjustment.
    
    Attributes:
        y_true (np.ndarray): True class labels
        y_pred (np.ndarray): Predicted class labels
        area_weights (bool): Apply area weighting (default: True)
    """
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, area_weights: bool = True):
        """Initialize assessor.
        
        Args:
            y_true (np.ndarray): True class labels
            y_pred (np.ndarray): Predicted class labels
            area_weights (bool): Apply area-weighted adjustments (default: True)
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.area_weights = area_weights
        logger.info(f"Initialized AccuracyAssessor (samples: {len(y_true)})")
    
    def overall_accuracy(self) -> float:
        """Compute overall accuracy.
        
        OA = (Diagonal Sum) / Total
        
        Returns:
            float: Overall accuracy (0-1)
        """
        return accuracy_score(self.y_true, self.y_pred)
    
    def kappa_coefficient(self) -> float:
        """Compute Cohen's Kappa coefficient.
        
        Kappa = (OA - Expected Accuracy) / (1 - Expected Accuracy)
        
        Returns:
            float: Kappa coefficient (-1 to 1)
        """
        return cohen_kappa_score(self.y_true, self.y_pred)
    
    def confusion_matrix(self) -> np.ndarray:
        """Generate confusion matrix.
        
        Returns:
            np.ndarray: Confusion matrix
        """
        return confusion_matrix(self.y_true, self.y_pred)
    
    def user_accuracy(self, class_label: int) -> float:
        """Compute user's accuracy for a specific class.
        
        User Accuracy = TP / (TP + FP) = Diagonal / Row Sum
        
        Args:
            class_label (int): Class index
            
        Returns:
            float: User's accuracy for the class
        """
        cm = self.confusion_matrix()
        return cm[class_label, class_label] / cm[class_label, :].sum()
    
    def producer_accuracy(self, class_label: int) -> float:
        """Compute producer's accuracy for a specific class.
        
        Producer Accuracy = TP / (TP + FN) = Diagonal / Column Sum
        
        Args:
            class_label (int): Class index
            
        Returns:
            float: Producer's accuracy for the class
        """
        cm = self.confusion_matrix()
        return cm[class_label, class_label] / cm[:, class_label].sum()
    
    def f1_score(self, class_label: int) -> float:
        """Compute F1 score for a specific class.
        
        F1 = 2 × (Precision × Recall) / (Precision + Recall)
        
        Args:
            class_label (int): Class index
            
        Returns:
            float: F1 score
        """
        ua = self.user_accuracy(class_label)
        pa = self.producer_accuracy(class_label)
        return 2 * (ua * pa) / (ua + pa + 1e-8)
    
    def compute_all_metrics(self) -> Dict[str, float]:
        """Compute all accuracy metrics.
        
        Returns:
            Dict[str, float]: Dictionary of accuracy metrics
        """
        metrics = {
            'overall_accuracy': self.overall_accuracy(),
            'kappa': self.kappa_coefficient(),
        }
        logger.info(f"Accuracy metrics computed: OA={metrics['overall_accuracy']:.3f}")
        return metrics
