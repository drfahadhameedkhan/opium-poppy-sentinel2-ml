"""
accuracy_assessment.py
=======================
Area-adjusted accuracy assessment for land cover classification.

Implements the Olofsson et al. (2014) protocol for computing area-adjusted
error matrices, overall accuracy, producer's and user's accuracy, Kappa
coefficient, and associated uncertainty estimates. Also implements McNemar's
test for statistical comparison between classifiers.

Author : Fahad Hameed Khan
Paper  : Geospatial Machine Learning for Opium Poppy Cultivation Monitoring
         in Pakistan and Afghanistan: A Sentinel-2 Multi-Temporal Analysis
Journal: Land (MDPI), 2025

References
----------
Olofsson et al. (2014) Remote Sensing of Environment, 148, 42–57.
Foody (2004) Photogrammetric Engineering & Remote Sensing, 70(5), 627–633.
McNemar (1947) Psychometrika, 12(2), 153–157.
"""

import numpy as np
from scipy import stats
import logging
from typing import Optional

logger = logging.getLogger(__name__)

CLASS_NAMES = ["Opium Poppy", "Wheat", "Bare/Fallow", "Other Vegetation", "Built/Water"]


class AccuracyAssessor:
    """
    Area-adjusted accuracy assessment (Olofsson et al. 2014).

    Computes unbiased area and accuracy estimates from a stratified
    random sample, accounting for the unequal spatial distribution of
    land cover classes.

    Parameters
    ----------
    y_true       : 1D array of true class labels
    y_pred       : 1D array of predicted class labels
    area_weights : 1D array of mapped class proportions (Wi), used for
                   area-adjusted statistics. If None, equal weights assumed.
    class_names  : list of class name strings
    """

    def __init__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        area_weights: Optional[np.ndarray] = None,
        class_names: Optional[list] = None,
    ):
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        self.n_classes = len(np.unique(self.y_true))
        self.class_names = class_names or CLASS_NAMES[: self.n_classes]
        self.area_weights = (
            area_weights if area_weights is not None
            else np.ones(self.n_classes) / self.n_classes
        )
        self._validate_inputs()

    def _validate_inputs(self):
        assert len(self.y_true) == len(self.y_pred), "y_true and y_pred must have same length."
        assert abs(self.area_weights.sum() - 1.0) < 1e-6, "area_weights must sum to 1."

    def confusion_matrix(self) -> np.ndarray:
        """Raw confusion matrix (rows=true, cols=predicted)."""
        from sklearn.metrics import confusion_matrix
        return confusion_matrix(self.y_true, self.y_pred)

    def area_adjusted_error_matrix(self) -> np.ndarray:
        """
        Compute area-adjusted error matrix (proportion of area).

        Converts the sample-count confusion matrix to proportional area
        estimates using the mapped area weights (Wi) per Olofsson et al. (2014).

        Returns
        -------
        q_matrix : np.ndarray, shape (n_classes, n_classes)
            p_ij = Wi * (n_ij / n_i) where n_ij = count in cell (i,j)
        """
        cm = self.confusion_matrix()
        q_matrix = np.zeros_like(cm, dtype=float)
        for i in range(self.n_classes):
            n_i = cm[i, :].sum()
            if n_i > 0:
                q_matrix[i, :] = self.area_weights[i] * cm[i, :] / n_i
        return q_matrix

    def overall_accuracy(self) -> Tuple[float, float]:
        """
        Area-adjusted overall accuracy ± 95% CI.

        OA = Σ p_ii
        Var(OA) = Σ Wi²(q_ii/n_i)(1 - q_ii/n_i) / (n_i - 1)

        Returns
        -------
        (OA, CI_95) as percentages
        """
        cm = self.confusion_matrix()
        q = self.area_adjusted_error_matrix()
        oa = np.trace(q)

        # Variance estimation (Olofsson et al. 2014, Eq. 5)
        var_oa = 0.0
        for i in range(self.n_classes):
            n_i = cm[i, :].sum()
            if n_i > 1:
                q_ii = cm[i, i] / n_i
                var_oa += (self.area_weights[i] ** 2) * q_ii * (1 - q_ii) / (n_i - 1)

        ci_95 = 1.96 * np.sqrt(var_oa)
        return oa * 100, ci_95 * 100

    def producers_accuracy(self) -> np.ndarray:
        """
        Area-adjusted producer's accuracy for each class.

        PA_j = p_jj / p_+j (column j marginal)
        Reflects omission error (missed detections — critical for poppy class).
        """
        q = self.area_adjusted_error_matrix()
        pa = np.zeros(self.n_classes)
        for j in range(self.n_classes):
            col_sum = q[:, j].sum()
            if col_sum > 0:
                pa[j] = q[j, j] / col_sum
        return pa * 100

    def users_accuracy(self) -> np.ndarray:
        """
        Area-adjusted user's accuracy for each class.

        UA_i = p_ii / p_i+ (row i marginal)
        Reflects commission error (false alarms).
        """
        q = self.area_adjusted_error_matrix()
        ua = np.zeros(self.n_classes)
        for i in range(self.n_classes):
            row_sum = q[i, :].sum()
            if row_sum > 0:
                ua[i] = q[i, i] / row_sum
        return ua * 100

    def f1_scores(self) -> np.ndarray:
        """Per-class F1 score from PA and UA."""
        pa = self.producers_accuracy() / 100
        ua = self.users_accuracy() / 100
        f1 = np.where(
            (pa + ua) > 0,
            2 * pa * ua / (pa + ua),
            0.0
        )
        return f1

    def kappa(self) -> float:
        """
        Cohen's Kappa coefficient of agreement.

        κ = (OA − Pe) / (1 − Pe)
        where Pe = Σ (p_i+ × p_+i) = expected agreement by chance.
        """
        q = self.area_adjusted_error_matrix()
        oa = np.trace(q)
        pe = sum(q[i, :].sum() * q[:, i].sum() for i in range(self.n_classes))
        if abs(1 - pe) < 1e-9:
            return 1.0
        return (oa - pe) / (1 - pe)

    def compute_all_metrics(self) -> dict:
        """
        Compute the full set of accuracy metrics and return as a dictionary.

        Returns
        -------
        dict with keys:
            overall_accuracy, kappa, producers_accuracy, users_accuracy,
            f1_scores, confusion_matrix, area_adjusted_matrix
        """
        oa, ci = self.overall_accuracy()
        results = {
            "overall_accuracy":     round(oa, 2),
            "overall_accuracy_ci":  round(ci, 2),
            "kappa":                round(self.kappa(), 4),
            "producers_accuracy":   {
                cls: round(pa, 2)
                for cls, pa in zip(self.class_names, self.producers_accuracy())
            },
            "users_accuracy": {
                cls: round(ua, 2)
                for cls, ua in zip(self.class_names, self.users_accuracy())
            },
            "f1_scores": {
                cls: round(f1, 4)
                for cls, f1 in zip(self.class_names, self.f1_scores())
            },
            "confusion_matrix":       self.confusion_matrix().tolist(),
            "area_adjusted_matrix":   self.area_adjusted_error_matrix().tolist(),
        }
        self._log_summary(results)
        return results

    def _log_summary(self, results: dict):
        logger.info("─" * 60)
        logger.info("ACCURACY ASSESSMENT SUMMARY (Olofsson et al. 2014)")
        logger.info("─" * 60)
        logger.info(f"  Overall Accuracy : {results['overall_accuracy']:.1f}% "
                    f"± {results['overall_accuracy_ci']:.1f}%")
        logger.info(f"  Kappa            : {results['kappa']:.4f}")
        logger.info(f"  Poppy PA         : {results['producers_accuracy'].get('Opium Poppy', 'N/A')}%")
        logger.info(f"  Poppy UA         : {results['users_accuracy'].get('Opium Poppy', 'N/A')}%")
        logger.info("─" * 60)

    def print_report(self):
        """Print a formatted accuracy report."""
        results = self.compute_all_metrics()
        print(f"\n{'═' * 65}")
        print(f"  CLASSIFICATION ACCURACY REPORT")
        print(f"{'═' * 65}")
        print(f"  Overall Accuracy : {results['overall_accuracy']:.1f}% ± {results['overall_accuracy_ci']:.1f}%")
        print(f"  Kappa            : {results['kappa']:.4f}")
        print(f"\n  Per-Class Metrics:")
        print(f"  {'Class':<22} {'PA (%)':>8}  {'UA (%)':>8}  {'F1':>8}")
        print(f"  {'─' * 50}")
        for cls in self.class_names:
            pa = results["producers_accuracy"].get(cls, 0)
            ua = results["users_accuracy"].get(cls, 0)
            f1 = results["f1_scores"].get(cls, 0)
            marker = " ◄" if cls == "Opium Poppy" else ""
            print(f"  {cls:<22} {pa:>8.1f}  {ua:>8.1f}  {f1:>8.4f}{marker}")
        print(f"{'═' * 65}\n")


# ── McNemar's test ────────────────────────────────────────────────────────────

def mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray,
                 y_pred_b: np.ndarray) -> dict:
    """
    McNemar's test for comparing two classifier predictions.

    Tests H₀: the two classifiers have equal error rates.
    Used to assess statistical significance of accuracy differences
    between CNN-LSTM, XGBoost, and Random Forest (Foody 2004).

    Parameters
    ----------
    y_true   : true class labels
    y_pred_a : predictions from classifier A
    y_pred_b : predictions from classifier B

    Returns
    -------
    dict with 'statistic', 'p_value', 'significant' (alpha=0.05)
    """
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # Discordant cells: A correct / B wrong, and A wrong / B correct
    e01 = np.sum(correct_a & ~correct_b)   # A right, B wrong
    e10 = np.sum(~correct_a & correct_b)   # A wrong, B right

    if e01 + e10 == 0:
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}

    # With continuity correction (Foody 2004)
    chi2 = (abs(e01 - e10) - 1) ** 2 / (e01 + e10)
    p_val = 1 - stats.chi2.cdf(chi2, df=1)

    result = {
        "statistic":   round(chi2, 3),
        "p_value":     round(p_val, 4),
        "significant": p_val < 0.05,
        "e01":         int(e01),
        "e10":         int(e10),
    }
    logger.info(f"McNemar test: χ²={chi2:.3f}, p={p_val:.4f} "
                f"({'significant' if result['significant'] else 'not significant'} at α=0.05)")
    return result


# ── Type hint fix for Python < 3.9 ───────────────────────────────────────────
from typing import Tuple
