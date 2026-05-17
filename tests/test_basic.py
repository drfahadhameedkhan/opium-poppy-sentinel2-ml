# tests/test_basic.py
import numpy as np

def test_ndvi_range():
    nir = np.array([0.5, 0.8, 0.3])
    red = np.array([0.1, 0.2, 0.4])
    ndvi = (nir - red) / (nir + red)
    assert np.all(ndvi >= -1) and np.all(ndvi <= 1)

def test_evi_positive():
    nir, red, blue = 0.6, 0.1, 0.05
    evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
    assert evi > 0

def test_cire_positive():
    red_edge3, red_edge1 = 0.4, 0.25
    cire = (red_edge3 / red_edge1) - 1
    assert cire > 0
