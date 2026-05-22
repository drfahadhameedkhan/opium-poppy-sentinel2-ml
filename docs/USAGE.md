# Usage Guide

## Quick Start

### 1. Data Preprocessing

```python
from src.preprocessing import Sentinel2Downloader

downloader = Sentinel2Downloader(
    start_date="2022-10-01",
    end_date="2023-05-31"
)
downloader.download(
    provinces=["Helmand", "Kandahar"],
    output_dir="data/processed/"
)
```

### 2. Feature Extraction

```python
from src.features import VegetationIndexExtractor

extractor = VegetationIndexExtractor()
ndvi = extractor.compute_ndvi(nir_band, red_band)
evi = extractor.compute_evi(nir_band, red_band, blue_band)
```

### 3. Model Training

```python
from src.models import CNNLSTMClassifier
import numpy as np

model = CNNLSTMClassifier(n_timesteps=26, n_features=13, n_classes=5)
model.fit(X_train, y_train, epochs=100, batch_size=256)
```

### 4. Evaluation

```python
from src.evaluation import AccuracyAssessor

assessor = AccuracyAssessor(y_true=y_test, y_pred=predictions)
print(f"OA: {assessor.overall_accuracy():.3f}")
print(f"Kappa: {assessor.kappa_coefficient():.3f}")
```

### 5. Map Generation

```python
from src.visualization import MapGenerator

mapper = MapGenerator(model=model, features=X_test, coordinates=coords)
mapper.generate_cultivation_map("results/map.tif")
```

---

## Complete Workflow

See Jupyter notebooks in `notebooks/` directory for end-to-end examples.
