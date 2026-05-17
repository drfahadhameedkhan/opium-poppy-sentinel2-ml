"""
vegetation_indices.py
=====================
Computes NDVI, EVI, and Red Edge Chlorophyll Index (CIre) time-series
from Sentinel-2 multi-temporal composites.

These three vegetation indices form the core spectral feature set for
phenologically discriminating opium poppy from spectrally similar licit crops
(primarily wheat) in the heterogeneous agricultural landscapes of
Afghanistan and Pakistan.

Author : Fahad Hameed Khan
Paper  : Geospatial Machine Learning for Opium Poppy Cultivation Monitoring
         in Pakistan and Afghanistan: A Sentinel-2 Multi-Temporal Analysis
Journal: Land (MDPI), 2025

References
----------
Tucker (1979) NDVI — Remote Sensing of Environment, 8(2), 127–150.
Huete et al. (2002) EVI — Remote Sensing of Environment, 83(1–2), 195–213.
Delegido et al. (2011) CIre — Sensors, 11(7), 7063–7081.
"""

import numpy as np
import logging
from typing import Union

logger = logging.getLogger(__name__)


class VegetationIndexExtractor:
    """
    Computes spectral vegetation indices from Sentinel-2 band arrays.

    Supports both numpy arrays (local processing) and Google Earth Engine
    images (cloud-based processing).

    Parameters
    ----------
    mode : str
        Processing mode: 'numpy' or 'gee' (default 'numpy').
    scale_factor : float
        Sentinel-2 L2A reflectance scaling factor (default 0.0001 for GEE).
    """

    def __init__(self, mode: str = "numpy", scale_factor: float = 0.0001):
        self.mode = mode
        self.scale_factor = scale_factor

    # ── NDVI ─────────────────────────────────────────────────────────────────

    def ndvi(
        self,
        nir: Union[np.ndarray, "ee.Image"],
        red: Union[np.ndarray, "ee.Image"],
    ) -> Union[np.ndarray, "ee.Image"]:
        """
        Normalised Difference Vegetation Index (Tucker, 1979).

        NDVI = (NIR − Red) / (NIR + Red)

        Sentinel-2 bands: NIR = B8 (842 nm), Red = B4 (665 nm).

        Parameters
        ----------
        nir : Band 8 reflectance (B8, 842 nm)
        red : Band 4 reflectance (B4, 665 nm)

        Returns
        -------
        NDVI array in range [−1, 1]
        """
        if self.mode == "gee":
            return nir.subtract(red).divide(nir.add(red)).rename("NDVI")
        # numpy
        nir = np.where(nir == 0, np.nan, nir) * self.scale_factor
        red = np.where(red == 0, np.nan, red) * self.scale_factor
        with np.errstate(invalid="ignore", divide="ignore"):
            ndvi = (nir - red) / (nir + red)
        return ndvi

    # ── EVI ──────────────────────────────────────────────────────────────────

    def evi(
        self,
        nir: Union[np.ndarray, "ee.Image"],
        red: Union[np.ndarray, "ee.Image"],
        blue: Union[np.ndarray, "ee.Image"],
        G: float = 2.5,
        C1: float = 6.0,
        C2: float = 7.5,
        L: float = 1.0,
    ) -> Union[np.ndarray, "ee.Image"]:
        """
        Enhanced Vegetation Index (Huete et al., 2002).

        EVI = G × (NIR − Red) / (NIR + C1×Red − C2×Blue + L)

        Sentinel-2 bands: NIR=B8, Red=B4, Blue=B2.
        Reduces atmospheric and soil background effects compared to NDVI.

        Parameters
        ----------
        nir  : Band 8 reflectance (NIR, 842 nm)
        red  : Band 4 reflectance (Red, 665 nm)
        blue : Band 2 reflectance (Blue, 490 nm)
        G    : Gain factor (default 2.5)
        C1   : Aerosol resistance coefficient 1 (default 6.0)
        C2   : Aerosol resistance coefficient 2 (default 7.5)
        L    : Canopy background adjustment (default 1.0)

        Returns
        -------
        EVI array (typically 0–1 for vegetated surfaces)
        """
        if self.mode == "gee":
            numerator = nir.subtract(red).multiply(G)
            denominator = nir.add(red.multiply(C1)).subtract(blue.multiply(C2)).add(L)
            return numerator.divide(denominator).rename("EVI")
        nir = nir * self.scale_factor
        red = red * self.scale_factor
        blue = blue * self.scale_factor
        with np.errstate(invalid="ignore", divide="ignore"):
            evi = G * (nir - red) / (nir + C1 * red - C2 * blue + L)
        return np.clip(evi, -1, 2)

    # ── CIre (Red Edge Chlorophyll Index) ────────────────────────────────────

    def cire(
        self,
        red_edge3: Union[np.ndarray, "ee.Image"],
        red_edge1: Union[np.ndarray, "ee.Image"],
    ) -> Union[np.ndarray, "ee.Image"]:
        """
        Red Edge Chlorophyll Index (Delegido et al., 2011).

        CIre = (Band7 / Band5) − 1

        Sentinel-2 bands: B7 = Red Edge 3 (783 nm), B5 = Red Edge 1 (705 nm).

        The red-edge spectral region is highly sensitive to chlorophyll content
        and crop stress. CIre provides discriminative information particularly
        relevant to distinguishing opium poppy from wheat during overlapping
        growth stages (February–March flowering window).

        Parameters
        ----------
        red_edge3 : Band 7 reflectance (783 nm)
        red_edge1 : Band 5 reflectance (705 nm)

        Returns
        -------
        CIre values (typically 0–5 for vegetated surfaces)
        """
        if self.mode == "gee":
            return red_edge3.divide(red_edge1).subtract(1).rename("CIre")
        red_edge3 = red_edge3 * self.scale_factor
        red_edge1 = np.where(red_edge1 == 0, np.nan, red_edge1) * self.scale_factor
        with np.errstate(invalid="ignore", divide="ignore"):
            cire = (red_edge3 / red_edge1) - 1
        return np.clip(cire, -1, 10)

    # ── Batch computation ─────────────────────────────────────────────────────

    def compute_all_indices(self, bands: dict) -> dict:
        """
        Compute all three vegetation indices from a band dictionary.

        Parameters
        ----------
        bands : dict
            Dictionary with keys 'NIR', 'Red', 'Blue', 'RedEdge1', 'RedEdge3'.
            Values are numpy arrays or ee.Images depending on self.mode.

        Returns
        -------
        dict with keys 'NDVI', 'EVI', 'CIre'.
        """
        logger.info("Computing vegetation indices: NDVI, EVI, CIre…")
        indices = {
            "NDVI": self.ndvi(bands["NIR"], bands["Red"]),
            "EVI":  self.evi(bands["NIR"], bands["Red"], bands["Blue"]),
            "CIre": self.cire(bands["RedEdge3"], bands["RedEdge1"]),
        }
        logger.info("✅ All vegetation indices computed.")
        return indices

    def compute_timeseries(self, timeseries_bands: list) -> dict:
        """
        Compute vegetation index time series from a sequence of band dictionaries.

        Parameters
        ----------
        timeseries_bands : list of (date_label: str, bands: dict)

        Returns
        -------
        dict of {index_name: np.ndarray of shape (n_timesteps, H, W)}
        """
        ndvi_series, evi_series, cire_series = [], [], []

        for date_label, bands in timeseries_bands:
            if bands is None:
                ndvi_series.append(None)
                evi_series.append(None)
                cire_series.append(None)
                continue
            indices = self.compute_all_indices(bands)
            ndvi_series.append(indices["NDVI"])
            evi_series.append(indices["EVI"])
            cire_series.append(indices["CIre"])

        return {
            "NDVI": ndvi_series,
            "EVI":  evi_series,
            "CIre": cire_series,
        }


# ── Utility functions ─────────────────────────────────────────────────────────

def smooth_timeseries(ts: np.ndarray, window: int = 3) -> np.ndarray:
    """
    Apply rolling median smoothing to a 1D time series.

    Parameters
    ----------
    ts     : 1D numpy array of index values across time
    window : smoothing window size (default 3 = 3 bi-weekly periods = 6 weeks)
    """
    from scipy.ndimage import median_filter
    return median_filter(ts, size=window, mode="reflect")


def fill_gaps_harmonic(ts: np.ndarray, n_harmonics: int = 3) -> np.ndarray:
    """
    Fill missing values (NaN) in a 1D time series using harmonic regression.

    Fits a Fourier series model to available observations and interpolates
    missing values (Jakubauskas et al. 2002; Zhou et al. 2016).

    Parameters
    ----------
    ts          : 1D numpy array with NaN gaps
    n_harmonics : number of harmonic components to fit (default 3)
    """
    t = np.arange(len(ts))
    valid = ~np.isnan(ts)

    if valid.sum() < 2 * n_harmonics + 1:
        # Not enough valid observations — linear interpolation fallback
        return np.interp(t, t[valid], ts[valid])

    # Build harmonic design matrix
    X = np.ones((valid.sum(), 1 + 2 * n_harmonics))
    t_valid = t[valid]
    for k in range(1, n_harmonics + 1):
        freq = 2 * np.pi * k / len(ts)
        X[:, 2 * k - 1] = np.cos(freq * t_valid)
        X[:, 2 * k]     = np.sin(freq * t_valid)

    # Fit harmonic model
    coeffs, _, _, _ = np.linalg.lstsq(X, ts[valid], rcond=None)

    # Predict for all time steps
    X_full = np.ones((len(ts), 1 + 2 * n_harmonics))
    for k in range(1, n_harmonics + 1):
        freq = 2 * np.pi * k / len(ts)
        X_full[:, 2 * k - 1] = np.cos(freq * t)
        X_full[:, 2 * k]     = np.sin(freq * t)

    fitted = X_full @ coeffs
    result = ts.copy()
    result[~valid] = fitted[~valid]
    return result
