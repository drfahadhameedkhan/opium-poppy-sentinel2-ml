"""
Tests for preprocessing module.
"""

import pytest
import numpy as np
from src.preprocessing import CloudMasker, Sentinel2Downloader, CompositeGenerator


class TestCloudMasking:
    """Test cloud masking functionality."""
    
    def test_cloud_masker_initialization(self):
        """Test CloudMasker initialization."""
        masker = CloudMasker(cloud_threshold=20)
        assert masker.cloud_threshold == 20
    
    def test_cloud_masker_default(self):
        """Test CloudMasker with default parameters."""
        masker = CloudMasker()
        assert masker.cloud_threshold == 20
    
    def test_scl_classes(self):
        """Test SCL class definitions."""
        masker = CloudMasker()
        scl_dict = masker.get_scl_classes()
        assert scl_dict['vegetation'] == 4
        assert scl_dict['cloud_high'] == 9


class TestSentinel2Downloader:
    """Test Sentinel-2 downloader."""
    
    def test_downloader_initialization(self):
        """Test Sentinel2Downloader initialization."""
        downloader = Sentinel2Downloader(
            start_date="2022-10-01",
            end_date="2023-05-31"
        )
        assert downloader.start_date == "2022-10-01"
        assert downloader.end_date == "2023-05-31"
    
    def test_collection_info(self):
        """Test collection info retrieval."""
        downloader = Sentinel2Downloader(
            start_date="2022-10-01",
            end_date="2023-05-31",
            cloud_threshold=15
        )
        info = downloader.get_collection_info()
        assert info['collection'] == 'COPERNICUS/S2_SR_HARMONIZED'
        assert info['cloud_threshold'] == 15


class TestCompositing:
    """Test composite generation."""
    
    def test_composite_generator_initialization(self):
        """Test CompositeGenerator initialization."""
        gen = CompositeGenerator(frequency="biweekly")
        assert gen.frequency == "biweekly"
    
    def test_composite_generator_monthly(self):
        """Test monthly compositing."""
        gen = CompositeGenerator(frequency="monthly")
        assert gen.frequency == "monthly"
