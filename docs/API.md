# API Documentation

## Overview

This document describes the public API for the Opium Poppy Sentinel-2 ML package.

## Modules

### `src.preprocessing`

Data acquisition and preprocessing from Sentinel-2 satellite imagery.

#### `Sentinel2Downloader`

Download Sentinel-2 L2A imagery from Google Earth Engine.

**Example:**
```python
from src.preprocessing import Sentinel2Downloader

downloader = Sentinel2Downloader(
    start_date="2022-10-01",
    end_date="2023-05-31",
    cloud_threshold=20
)
downloader.download(
    provinces=["Helmand", "Kandahar"],
    output_dir="data/processed/"
)
```

#### `CloudMasker`

Remove clouds, shadows, and defective pixels using SCL layer.

**Example:**
```python
from src.preprocessing import CloudMasker

masker = CloudMasker(cloud_threshold=20)
masked_image = masker.mask_clouds(sentinel2_image)
```

---

### `src.features`

Feature extraction from satellite imagery.

#### `VegetationIndexExtractor`

Compute vegetation indices (NDVI, EVI, CIre).

**Example:**
```python
from src.features import VegetationIndexExtractor
import numpy as np

extractor = VegetationIndexExtractor()
ndvi = extractor.compute_ndvi(nir=nir_band, red=red_band)
evi = extractor.compute_evi(nir=nir_band, red=red_band, blue=blue_band)
```

---

### `src.models`

Machine learning classifiers for cultivation detection.

#### `RandomForestClassifier`

Random Forest with 500 trees for classification.

**Example:**
```python
from src.models import RandomForestClassifier

model = RandomForestClassifier(n_estimators=500, max_depth=20)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

#### `XGBoostClassifier`

Gradient Boosting classifier.

**Example:**
```python
from src.models import XGBoostClassifier

model = XGBoostClassifier(n_estimators=500, learning_rate=0.1)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

#### `CNNLSTMClassifier`

Hybrid CNN-LSTM model for temporal classification.

**Example:**
```python
from src.models import CNNLSTMClassifier

model = CNNLSTMClassifier(
    n_timesteps=26,
    n_features=13,
    n_classes=5
)

history = model.fit(X_train, y_train, epochs=100)
predictions = model.predict(X_test)
```

---

### `src.evaluation`

Model evaluation and accuracy assessment.

#### `AccuracyAssessor`

Compute accuracy metrics.

**Example:**
```python
from src.evaluation import AccuracyAssessor

assessor = AccuracyAssessor(y_true=y_test, y_pred=predictions)
print(f"OA: {assessor.overall_accuracy():.3f}")
print(f"Kappa: {assessor.kappa_coefficient():.3f}")
```

---

## Configuration Files

See `configs/` directory for model and GEE configurations.
