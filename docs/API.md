# API Documentation

## `src.preprocessing.Sentinel2Preprocessor`

### Methods

#### `create_biweekly_composites()`
Creates bi-weekly median composites from Sentinel-2 imagery.

**Parameters:**
- `start_date` (str): Start date in YYYY-MM-DD format
- `end_date` (str): End date in YYYY-MM-DD format
- `cloud_threshold` (float): Max cloud percentage (default: 20)

**Returns:**
- `GeoTIFF`: Composite imagery

**Example:**
```python
preprocessor = Sentinel2Preprocessor("2022-10-01", "2023-05-31")
composites = preprocessor.create_biweekly_composites()
