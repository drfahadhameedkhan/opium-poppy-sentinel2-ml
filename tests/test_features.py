"""
Tests for feature extraction module.
"""

import pytest
import numpy as np
from src.features import VegetationIndexExtractor, PhenologicalExtractor, TerrainExtractor


class TestVegetationIndices:
    """Test vegetation index computation."""
    
    def setup_method(self):
        """Setup test data."""
        self.extractor = VegetationIndexExtractor()
        self.nir = np.array([0.5, 0.6, 0.7])
        self.red = np.array([0.1, 0.2, 0.3])
        self.blue = np.array([0.05, 0.1, 0.15])
    
    def test_ndvi_computation(self):
        """Test NDVI calculation."""
        ndvi = self.extractor.compute_ndvi(self.nir, self.red)
        assert ndvi.shape == self.nir.shape
        assert np.all((ndvi >= -1) & (ndvi <= 1))  # NDVI range
    
    def test_ndvi_values(self):
        """Test NDVI produces expected values."""
        nir = np.array([0.5])
        red = np.array([0.2])
        ndvi = self.extractor.compute_ndvi(nir, red)
        expected = (0.5 - 0.2) / (0.5 + 0.2)
        assert np.isclose(ndvi[0], expected, atol=1e-6)
    
    def test_evi_computation(self):
        """Test EVI calculation."""
        evi = self.extractor.compute_evi(self.nir, self.red, self.blue)
        assert evi.shape == self.nir.shape
    
    def test_cire_computation(self):
        """Test CIre calculation."""
        band7 = np.array([0.8, 0.9, 1.0])
        band5 = np.array([0.7, 0.8, 0.9])
        cire = self.extractor.compute_cire(band7, band5)
        assert cire.shape == band7.shape


class TestPhenologicalExtractor:
    """Test phenological parameter extraction."""
    
    def setup_method(self):
        """Setup test data."""
        self.extractor = PhenologicalExtractor(n_harmonics=3)
        self.ts_data = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.7, 0.5, 0.3, 0.2])
    
    def test_initialization(self):
        """Test initialization."""
        assert self.extractor.n_harmonics == 3
    
    def test_fit_harmonic_model(self):
        """Test harmonic model fitting."""
        params = self.extractor.fit_harmonic_model(self.ts_data)
        assert 'mean' in params
        assert 'amplitude' in params
        assert 'phase' in params
    
    def test_peak_date(self):
        """Test peak date extraction."""
        peak = self.extractor.get_peak_date(self.ts_data)
        assert peak >= 0
        assert peak < len(self.ts_data)


class TestTerrainExtractor:
    """Test terrain feature extraction."""
    
    def test_initialization(self):
        """Test TerrainExtractor initialization."""
        extractor = TerrainExtractor()
        assert extractor is not None
