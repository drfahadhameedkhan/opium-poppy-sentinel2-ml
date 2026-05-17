"""
sentinel2_download.py
=====================
Google Earth Engine-based Sentinel-2 Level-2A acquisition pipeline.

Acquires multi-temporal Sentinel-2 imagery for opium poppy cultivation
monitoring in Afghanistan and Pakistan study provinces.

Author : Fahad Hameed Khan
Paper  : Geospatial Machine Learning for Opium Poppy Cultivation Monitoring
         in Pakistan and Afghanistan: A Sentinel-2 Multi-Temporal Analysis
Journal: Land (MDPI), 2025
"""

import ee
import json
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Study province definitions ────────────────────────────────────────────────

STUDY_PROVINCES = {
    "afghanistan": {
        "Helmand":     ee.Geometry.Rectangle([61.5, 29.5, 65.5, 32.5]),
        "Kandahar":    ee.Geometry.Rectangle([64.0, 30.5, 67.5, 32.5]),
        "Nangarhar":   ee.Geometry.Rectangle([69.5, 33.5, 71.5, 35.0]),
        "Uruzgan":     ee.Geometry.Rectangle([65.0, 32.0, 67.0, 34.0]),
        "Badakhshan":  ee.Geometry.Rectangle([70.0, 36.0, 75.0, 38.5]),
    },
    "pakistan": {
        "KPK":         ee.Geometry.Rectangle([70.0, 33.0, 74.0, 36.5]),
        "Balochistan": ee.Geometry.Rectangle([62.0, 26.0, 68.0, 31.5]),
    }
}

# Sentinel-2 bands used in feature engineering (Immitzer et al. 2016)
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
S2_BAND_NAMES = ["Blue", "Green", "Red", "RedEdge1", "RedEdge2",
                 "RedEdge3", "NIR", "NarrowNIR", "SWIR1", "SWIR2"]

# Cultivation season windows (Oct → May)
CULTIVATION_SEASONS = {
    "S1": ("2022-10-01", "2023-05-31"),
    "S2": ("2023-10-01", "2024-05-31"),
    "S3": ("2024-10-01", "2025-05-31"),
}


class Sentinel2Preprocessor:
    """
    End-to-end Sentinel-2 preprocessing pipeline.

    Acquires Sentinel-2 Level-2A imagery, applies SCL-based cloud masking,
    and creates bi-weekly median composites for phenological feature extraction.

    Parameters
    ----------
    start_date : str
        ISO format start date (e.g., '2022-10-01').
    end_date : str
        ISO format end date (e.g., '2023-05-31').
    provinces : list[str]
        Province names to process (see STUDY_PROVINCES).
    cloud_threshold : int
        Maximum cloud cover percentage per scene (default 20).
    composite_days : int
        Composite window in days (default 14 = bi-weekly).
    output_dir : str or Path
        Directory for exported GeoTIFF composites.
    """

    def __init__(
        self,
        start_date: str,
        end_date: str,
        provinces: list,
        cloud_threshold: int = 20,
        composite_days: int = 14,
        output_dir: Optional[str] = None,
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.provinces = provinces
        self.cloud_threshold = cloud_threshold
        self.composite_days = composite_days
        self.output_dir = Path(output_dir) if output_dir else Path("data/processed")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._init_gee()

    def _init_gee(self):
        """Authenticate and initialise the Earth Engine session."""
        try:
            ee.Initialize()
            logger.info("Google Earth Engine initialised successfully.")
        except Exception:
            logger.warning("EE not initialised — attempting authentication.")
            ee.Authenticate()
            ee.Initialize()

    def _mask_clouds_scl(self, image: ee.Image) -> ee.Image:
        """
        Apply SCL-based cloud masking.

        Pixels classified as cloud (8,9), cloud shadow (3), saturated (1),
        or defective (2) are excluded (Griffiths et al. 2019).
        """
        scl = image.select("SCL")
        valid_mask = (
            scl.neq(1)   # saturated/defective
            .And(scl.neq(2))   # dark area
            .And(scl.neq(3))   # cloud shadow
            .And(scl.neq(8))   # cloud medium prob
            .And(scl.neq(9))   # cloud high prob
            .And(scl.neq(10))  # thin cirrus
        )
        return image.updateMask(valid_mask)

    def _get_province_geometry(self, province_name: str) -> ee.Geometry:
        """Retrieve GEE geometry for a named province."""
        for country_provinces in STUDY_PROVINCES.values():
            if province_name in country_provinces:
                return country_provinces[province_name]
        raise ValueError(
            f"Province '{province_name}' not found. "
            f"Available: {list(STUDY_PROVINCES['afghanistan'].keys()) + list(STUDY_PROVINCES['pakistan'].keys())}"
        )

    def _build_collection(self, geometry: ee.Geometry) -> ee.ImageCollection:
        """
        Build a filtered, cloud-masked Sentinel-2 collection.

        Returns Sentinel-2 SR collection with SCL masking applied.
        """
        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(geometry)
            .filterDate(self.start_date, self.end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", self.cloud_threshold))
            .select(S2_BANDS + ["SCL"], S2_BAND_NAMES + ["SCL"])
            .map(self._mask_clouds_scl)
        )
        n_images = collection.size().getInfo()
        logger.info(f"  → {n_images} cloud-filtered scenes acquired.")
        return collection

    def _create_biweekly_composites(
        self, collection: ee.ImageCollection, geometry: ee.Geometry
    ) -> list:
        """
        Create bi-weekly median composites.

        For each 14-day window within the season, computes a median composite
        across available cloud-free observations. Windows with > 80% cloud
        coverage are flagged for harmonic gap-filling.

        Returns a list of (date_label, ee.Image) tuples.
        """
        import pandas as pd

        date_range = pd.date_range(self.start_date, self.end_date, freq=f"{self.composite_days}D")
        composites = []

        for i, start in enumerate(date_range[:-1]):
            end = date_range[i + 1]
            window_label = start.strftime("%Y-%m-%d")

            window_col = collection.filterDate(
                start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
            )
            n_obs = window_col.size().getInfo()

            if n_obs == 0:
                logger.warning(f"  ⚠ No observations for window {window_label} — flagged for gap-fill.")
                composites.append((window_label, None))
            else:
                composite = window_col.median().clip(geometry)
                composite = composite.set("system:time_start",
                                         ee.Date(start.strftime("%Y-%m-%d")).millis())
                composites.append((window_label, composite))
                logger.debug(f"  ✓ Composite {window_label}: {n_obs} scenes.")

        return composites

    def process_province(self, province_name: str) -> list:
        """
        Full preprocessing pipeline for a single province.

        Parameters
        ----------
        province_name : str
            Province to process.

        Returns
        -------
        list of (date_label, ee.Image)
            Bi-weekly cloud-free composites.
        """
        logger.info(f"Processing province: {province_name}")
        geometry = self._get_province_geometry(province_name)
        collection = self._build_collection(geometry)
        composites = self._create_biweekly_composites(collection, geometry)
        logger.info(f"  → {len([c for c in composites if c[1] is not None])} valid composites "
                    f"/ {len(composites)} windows for {province_name}.")
        return composites

    def export_composites(self, province_name: str, composites: list, scale: int = 10):
        """
        Export composites to Google Drive as GeoTIFF files.

        Parameters
        ----------
        province_name : str
        composites    : list of (label, ee.Image)
        scale         : int — spatial resolution in metres (default 10)
        """
        logger.info(f"Exporting {len(composites)} composites for {province_name}…")
        for label, image in composites:
            if image is None:
                continue
            task = ee.batch.Export.image.toDrive(
                image=image,
                description=f"S2_{province_name}_{label}",
                folder="Sentinel2_Poppy_Monitor",
                fileNamePrefix=f"S2_{province_name}_{label}",
                scale=scale,
                maxPixels=1e13,
                fileFormat="GeoTIFF",
            )
            task.start()
            logger.info(f"  → Export task started: S2_{province_name}_{label}")

    def run(self, export: bool = False):
        """
        Run the full preprocessing pipeline for all specified provinces.

        Parameters
        ----------
        export : bool — if True, export composites to Google Drive.

        Returns
        -------
        dict — {province_name: [(label, ee.Image), ...]}
        """
        all_composites = {}
        for province in self.provinces:
            composites = self.process_province(province)
            all_composites[province] = composites
            if export:
                self.export_composites(province, composites)
        logger.info("✅ Preprocessing pipeline complete.")
        return all_composites


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sentinel-2 preprocessing pipeline for opium poppy monitoring."
    )
    parser.add_argument("--start_date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--provinces", nargs="+", required=True, help="Province names")
    parser.add_argument("--cloud_threshold", type=int, default=20)
    parser.add_argument("--output_dir", default="data/processed/")
    parser.add_argument("--export", action="store_true", help="Export to Google Drive")
    args = parser.parse_args()

    preprocessor = Sentinel2Preprocessor(
        start_date=args.start_date,
        end_date=args.end_date,
        provinces=args.provinces,
        cloud_threshold=args.cloud_threshold,
        output_dir=args.output_dir,
    )
    preprocessor.run(export=args.export)
