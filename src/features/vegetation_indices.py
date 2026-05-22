"""
Compute vegetation indices (NDVI, EVI, CIre).

Calculates spectral indices from Sentinel-2 bands for vegetation monitoring.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class VegetationIndexExtractor:
    """Extract vegetation indices from spectral bands.
    
    Implements standard vegetation indices used in remote sensing:
    - NDVI: Normalized Difference Vegetation Index
    - EVI: Enhanced Vegetation Index
    - CIre: Chlorophyll Index Red Edge
    """
    
    def __init__(self):
        """Initialize vegetation index extractor."""
        logger.info("Initialized VegetationIndexExtractor")
    
    def compute_ndvi(self, nir: np.ndarray, red: np.ndarray) -> np.ndarray:
        """Compute Normalized Difference Vegetation Index (NDVI).
        
        Formula: (NIR - Red) / (NIR + Red)
        
        Args:
            nir (np.ndarray): Near-infrared band (Band 8)
            red (np.ndarray): Red band (Band 4)
            
        Returns:
            np.ndarray: NDVI values in range [-1, 1]
        """
        return (nir - red) / (nir + red + 1e-8)
    
    def compute_evi(self, nir: np.ndarray, red: np.ndarray, blue: np.ndarray) -> np.ndarray:
        """Compute Enhanced Vegetation Index (EVI).
        
        Formula: 2.5 × (NIR - Red) / (NIR + 6×Red - 7.5×Blue + 1)
        
        Args:
            nir (np.ndarray): Near-infrared band (Band 8)
            red (np.ndarray): Red band (Band 4)
            blue (np.ndarray): Blue band (Band 2)
            
        Returns:
            np.ndarray: EVI values in range [-1, 1]
        """
        return 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + 1e-8)
    
    def compute_cire(self, band7: np.ndarray, band5: np.ndarray) -> np.ndarray:
        """Compute Chlorophyll Index Red Edge (CIre).
        
        Formula: (Band7 / Band5) - 1
        
        Uses Red Edge bands for chlorophyll content estimation.
        Useful for discriminating flowering stages.
        
        Args:
            band7 (np.ndarray): Red Edge band (783 nm)
            band5 (np.ndarray): Red Edge band (740 nm)
            
        Returns:
            np.ndarray: CIre values
        """
        return (band7 / (band5 + 1e-8)) - 1
