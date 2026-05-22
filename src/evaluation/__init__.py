"""
Model evaluation and accuracy assessment module.

Computes accuracy metrics, confusion matrices, and spatial cross-validation.
"""

from .accuracy_assessment import AccuracyAssessor
from .spatial_cv import SpatialBlockCV

__all__ = [
    "AccuracyAssessor",
    "SpatialBlockCV",
]
