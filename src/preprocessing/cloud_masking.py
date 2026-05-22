"""
SCL-based cloud masking for Sentinel-2 imagery.

Removes clouds, shadows, and defective pixels using Scene Classification Layer (SCL).
"""

import logging
import ee

logger = logging.getLogger(__name__)


class CloudMasker:
    """Remove clouds, shadows, and defective pixels using SCL layer.
    
    The Scene Classification Layer (SCL) provides pixel-level classification:
    - 0: No Data
    - 1: Saturated / Defective
    - 2: Dark Area Pixels
    - 3: Cloud Shadows
    - 4: Vegetation
    - 5: Non-Vegetated
    - 6: Water
    - 7: Unclassified
    - 8: Cloud Medium Probability
    - 9: Cloud High Probability
    - 10: Thin Cirrus
    - 11: Snow / Ice
    """
    
    def __init__(self, cloud_threshold: int = 20):
        """Initialize cloud masker.
        
        Args:
            cloud_threshold (int): Cloud percentage threshold (default: 20%)
        """
        self.cloud_threshold = cloud_threshold
        logger.info(f"Initialized CloudMasker with threshold {cloud_threshold}%")
    
    def mask_clouds(self, image: ee.Image) -> ee.Image:
        """Apply cloud mask to image.
        
        Args:
            image (ee.Image): Input Sentinel-2 image
            
        Returns:
            ee.Image: Masked image with clouds and shadows removed
        """
        scl = image.select('SCL')
        # Clouds: 8, 9; Shadows: 3; Poor quality: 1
        cloud_mask = scl.neq(8).And(scl.neq(9)).And(scl.neq(3)).And(scl.neq(1))
        return image.updateMask(cloud_mask)
