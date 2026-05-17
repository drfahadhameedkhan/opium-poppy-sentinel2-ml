# 🌍 Geospatial Machine Learning for Opium Poppy Cultivation Monitoring
## A Sentinel-2 Multi-Temporal Analysis - Pakistan & Afghanistan

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://earthengine.google.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.xxxx%2Fland.2025.xxxxx-blue?style=for-the-badge)](https://doi.org)
[![Acceptance Awaited from Journal](https://img.shields.io/badge/Journal-Land%20(MDPI)-orange?style=for-the-badge)](https://www.mdpi.com/journal/land)

[![Stars](https://img.shields.io/github/stars/fahadhameedkhan/opium-poppy-sentinel2-ml?style=social)](https://github.com/fahadhameedkhan/opium-poppy-sentinel2-ml/stargazers)
[![Forks](https://img.shields.io/github/forks/fahadhameedkhan/opium-poppy-sentinel2-ml?style=social)](https://github.com/fahadhameedkhan/opium-poppy-sentinel2-ml/network)

</div>

---

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   SENTINEL-2  ──►  FEATURE ENGINEERING  ──►  ML CLASSIFICATION         │
│    (4,872 tiles)     (NDVI · EVI · CIre)    (RF · XGBoost · CNN-LSTM)  │
│                                                                         │
│   OA = 91.4%  │  Kappa = 0.88  │  680,000 km²  │  3 Seasons (2022–25) │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Results](#-key-results)
- [Study Area](#-study-area)
- [Methodology](#-methodology)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Modules](#-modules)
- [Results & Figures](#-results--figures)
- [Data Availability](#-data-availability)
- [Citation](#-citation)
- [Author](#-author)

---

## 🔬 Overview

This repository contains the **complete open-source implementation** of a geospatial machine learning framework for **detecting, mapping and monitoring illicit opium poppy cultivation** in South Asia using freely available multi-temporal **Sentinel-2** satellite imagery.

The study develops and rigorously validates three supervised classification architectures: **Random Forest (RF)**, **XGBoost** and a hybrid **Convolutional Neural Network–Long Short-Term Memory (CNN-LSTM)**, for cultivation area estimation across **680,000 km²** spanning major producer provinces in **Afghanistan and Pakistan** over three consecutive cultivation seasons (2022–2025).

> **Manuscript submitted for acceptance to:** *Land* (MDPI)

### ✨ What Makes This Framework Different?

| Feature | This Study | Prior Studies |
|---|---|---|
| **Deep Learning (CNN-LSTM)** | ✅ First applied to South Asia | ❌ Not used |
| **Dual-country coverage** | ✅ Afghanistan + Pakistan | ❌ Afghanistan only |
| **UNODC validation** | ✅ National-level benchmark | ❌ Internal validation |
| **Risk surface modelling** | ✅ 18,400 km² identified | ❌ Not included |
| **Open-source pipelines** | ✅ Fully released | ❌ Code not shared |
| **Overall Accuracy** | ✅ **91.4%** | ⬇ 68–91% |

---

## 📊 Key Results

<div align="center">

### Classification Performance

| Classifier | Overall Accuracy | Kappa | PA - Poppy | UA - Poppy | F1 Score |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 🌳 **Random Forest** | 87.3 ± 2.6% | 0.83 ± 0.03 | 83.1 ± 3.1% | 81.7 ± 2.9% | 0.824 |
| ⚡ **XGBoost** | 89.7 ± 2.1% | 0.86 ± 0.02 | 86.4 ± 2.4% | 85.2 ± 2.7% | 0.858 |
| 🧠 **CNN-LSTM** *(Best)* | **91.4 ± 1.8%** | **0.88 ± 0.02** | **89.2 ± 1.9%** | **87.6 ± 2.1%** | **0.884** |

### Cultivation Area Estimates vs. UNODC Benchmarks

| Season | UNODC (ha) | Satellite (ha) | Relative Error | r² |
|:---:|:---:|:---:|:---:|:---:|
| 2022–23 Afghanistan | 233,000 | 225,530 | −3.2% | 0.94 |
| 2023–24 Afghanistan | 296,400 | 313,480 | +5.8% | 0.93 |
| 2024–25 Afghanistan | 241,700 | 259,560 | +7.4% | 0.91 |
| 2022–23 KPK, Pakistan | 1,460 | 1,370 | −6.2% | 0.87 |
| 2022–23 Balochistan | 820 | 890 | +8.5% | 0.84 |

### Provincial Change Detection (Season 1 → Season 3)

```
Helmand (AFG)    ████████████████████████████████████  +14,200 ha  (+11.6%)
Kandahar (AFG)   ████████████                          + 3,600 ha  ( +9.4%)
Uruzgan (AFG)    ████                                  + 1,300 ha  ( +7.2%)
KPK (PAK)        ██                                    +   180 ha  (+13.1%)
Balochistan (PAK)██                                    +   130 ha  (+14.6%)
Badakhshan (AFG) ░░                                    -   600 ha  ( -4.9%)
Nangarhar (AFG)  ░░░░░░░░░░                            - 6,800 ha  (-23.7%)
```

</div>

---

## 🗺️ Study Area

```
                    STUDY REGION: South Asia (~680,000 km²)
    ┌─────────────────────────────────────────────────────────┐
    │                                                         │
    │   AFGHANISTAN                    PAKISTAN               │
    │   ──────────                     ───────                │
    │   • Helmand    (122,400 ha)      • KPK/FATA  (1,460 ha)│
    │   • Kandahar   ( 38,200 ha)      • Balochistan( 820 ha)│
    │   • Nangarhar  ( 28,700 ha)                             │
    │   • Uruzgan    ( 18,100 ha)                             │
    │   • Badakhshan ( 12,200 ha)                             │
    │                                                         │
    │   Elevation range: 600 – 4,000 m                        │
    │   Climate: Arid–Semi-arid, Highland                     │
    │   Total Sentinel-2 tiles: 4,872                         │
    └─────────────────────────────────────────────────────────┘
```

The study encompasses diverse agro-ecological zones:
- 🏜️ **Irrigated lowlands** - Helmand/Kandahar river valleys (750–1,200 m)
- 🏔️ **Highland terraces** - KPK Bara, Tirah, Dir valleys (steep slopes)
- 🌾 **Mid-elevation rain-fed fields** - Nangarhar, Badakhshan (600–4,000 m)
- 🌵 **Arid highland plateaux** - Northern/western Balochistan

---

## 🔧 Methodology

```
DATA ACQUISITION          PREPROCESSING             FEATURE ENGINEERING
──────────────────        ─────────────────         ───────────────────
Sentinel-2 L2A    ──►    Cloud Masking (SCL) ──►   NDVI time-series
4,872 tiles               Bi-weekly composites      EVI time-series
Oct 2022–May 2025         Harmonic interpolation     Red Edge CIre
Google Earth Engine       10 bands selected          Phenological params
                                                     SRTM terrain covars
         │
         ▼
TRAINING DATA             CLASSIFICATION            ACCURACY ASSESSMENT
──────────────────        ─────────────────         ───────────────────
38,417 labelled px ──►   Random Forest      ──►   Area-adjusted OA
UNODC survey pts          XGBoost                   Kappa coefficient
WorldView-3 VHR           CNN-LSTM (best)            McNemar's test
SMOTE oversampling         TensorFlow 2.x            Spatial block CV
Spatial block CV           scikit-learn / xgb        UNODC validation
         │
         ▼
CHANGE DETECTION          RISK SURFACE MODEL        POLICY OUTPUTS
──────────────────        ─────────────────         ───────────────────
Bi-temporal diff  ──►    Logistic regression ──►   Cultivation maps
2022–23 → 2024–25         AUC = 0.89                Risk surface (4 tiers)
District aggregation       Elastic-net reg           18,400 km² flagged
UNODC trend validation     Terrain + infra covars    Open-source tools
```

### Vegetation Indices

| Index | Formula | Purpose |
|---|---|---|
| **NDVI** | `(NIR − Red) / (NIR + Red)` | Canopy density & greenness |
| **EVI** | `2.5 × (NIR − Red) / (NIR + 6×Red − 7.5×Blue + 1)` | High-biomass canopy, reduced soil effects |
| **CIre** | `(Band7 / Band5) − 1` | Chlorophyll content, flowering-stage discrimination |

### CNN-LSTM Architecture

```
Input (26 timesteps × 13 features)
    │
    ▼
Conv1D(64, kernel=3) + BatchNorm + Dropout(0.3)
    │
    ▼
Conv1D(128, kernel=3) + BatchNorm + Dropout(0.3)
    │
    ▼
MaxPooling1D
    │
    ▼
LSTM(128) + Dropout(0.2)
    │
    ▼
LSTM(128) + Dropout(0.2)
    │
    ▼
Dense(64, ReLU)
    │
    ▼
Softmax(5 classes)  →  [Poppy | Wheat | Bare | Vegetation | Built/Water]
```

---

## 📁 Repository Structure

```
opium-poppy-sentinel2-ml/
│
├── 📄 README.md                    ← You are here
├── 📄 LICENSE                      ← MIT License
├── 📄 CITATION.cff                 ← Machine-readable citation
├── 📄 requirements.txt             ← Python dependencies
├── 📄 environment.yml              ← Conda environment
├── 📄 setup.py                     ← Package setup
│
├── 📂 src/                         ← Core source code
│   ├── 📂 preprocessing/
│   │   ├── sentinel2_download.py   ← GEE data acquisition
│   │   ├── cloud_masking.py        ← SCL-based cloud masking
│   │   └── compositing.py         ← Bi-weekly median composites
│   │
│   ├── 📂 features/
│   │   ├── vegetation_indices.py   ← NDVI, EVI, CIre computation
│   │   ├── phenological_params.py  ← Harmonic regression phenology
│   │   └── terrain_derivatives.py ← SRTM-based terrain features
│   │
│   ├── 📂 models/
│   │   ├── random_forest.py        ← RF classifier + hyperopt
│   │   ├── xgboost_classifier.py   ← XGBoost + Bayesian search
│   │   └── cnn_lstm.py             ← Hybrid CNN-LSTM architecture
│   │
│   ├── 📂 evaluation/
│   │   ├── accuracy_assessment.py  ← Area-adjusted error matrices
│   │   └── spatial_cv.py           ← Spatial block cross-validation
│   │
│   └── 📂 visualization/
│       └── maps.py                 ← Cultivation maps & risk surfaces
│
├── 📂 notebooks/                   ← Jupyter workflows (end-to-end)
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_accuracy_assessment.ipynb
│   └── 05_change_detection_risk.ipynb
│
├── 📂 configs/
│   ├── gee_config.yaml             ← GEE parameters
│   └── model_config.yaml          ← Model hyperparameters
│
├── 📂 data/
│   ├── README.md                   ← Data access instructions
│   └── sample/                    ← Sample data for testing
│
└── 📂 results/
    ├── figures/                    ← Output maps and plots
    └── README.md                   ← Results description
```

---

## ⚙️ Installation

### Prerequisites

- Python ≥ 3.9
- Google Earth Engine account (free registration at [earthengine.google.com](https://earthengine.google.com))
- ≥ 16 GB RAM recommended for CNN-LSTM training
- GPU optional but recommended for deep learning

### Step 1 — Clone the Repository

```bash
git clone https://github.com/fahadhameedkhan/opium-poppy-sentinel2-ml.git
cd opium-poppy-sentinel2-ml
```

### Step 2 — Create the Conda Environment (Recommended)

```bash
conda env create -f environment.yml
conda activate poppy-monitor
```

### Step 3 — Alternative: pip install

```bash
pip install -r requirements.txt
```

### Step 4 — Authenticate Google Earth Engine

```bash
earthengine authenticate
```

### Step 5 — Verify Installation

```bash
python -c "import src; print('✅ All modules loaded successfully')"
```

---

## 🚀 Quick Start

### End-to-End Pipeline (Command Line)

```bash
# Step 1: Download and preprocess Sentinel-2 imagery
python src/preprocessing/sentinel2_download.py \
    --start_date 2022-10-01 \
    --end_date 2023-05-31 \
    --provinces helmand kandahar nangarhar \
    --output_dir data/processed/

# Step 2: Compute vegetation indices and phenological features
python src/features/vegetation_indices.py \
    --input_dir data/processed/ \
    --output_dir data/features/

python src/features/phenological_params.py \
    --input_dir data/features/ \
    --n_harmonics 3

# Step 3: Train classifiers
python src/models/cnn_lstm.py \
    --training_data data/features/train_samples.csv \
    --model_output results/models/cnn_lstm_best.h5 \
    --epochs 100 --batch_size 256

# Step 4: Generate cultivation maps
python src/visualization/maps.py \
    --model results/models/cnn_lstm_best.h5 \
    --features data/features/season3/ \
    --output results/figures/cultivation_map_2024_25.tif
```

### Python API

```python
from src.preprocessing import Sentinel2Preprocessor
from src.features import VegetationIndexExtractor, PhenologicalExtractor
from src.models import CNNLSTMClassifier
from src.evaluation import AccuracyAssessor

# 1. Preprocess Sentinel-2 imagery
preprocessor = Sentinel2Preprocessor(
    start_date="2022-10-01",
    end_date="2023-05-31",
    provinces=["Helmand", "Kandahar"],
    cloud_threshold=20
)
composites = preprocessor.create_biweekly_composites()

# 2. Extract features
vi_extractor = VegetationIndexExtractor(composites)
features = vi_extractor.compute_all_indices()  # NDVI, EVI, CIre

pheno_extractor = PhenologicalExtractor(features["NDVI"])
pheno_params = pheno_extractor.fit_harmonic_model(n_harmonics=3)

# 3. Train CNN-LSTM
classifier = CNNLSTMClassifier(
    n_timesteps=26,
    n_features=13,
    n_classes=5,
    dropout_rate=0.3
)
classifier.fit(X_train, y_train, epochs=100, batch_size=256)

# 4. Assess accuracy
assessor = AccuracyAssessor(classifier, X_test, y_test, area_weights=True)
results = assessor.compute_all_metrics()
print(f"OA: {results['overall_accuracy']:.1f}%  |  Kappa: {results['kappa']:.3f}")
```

---

## 📦 Modules

### `src/preprocessing/`

| Module | Description |
|---|---|
| `sentinel2_download.py` | GEE-based Sentinel-2 L2A acquisition for study provinces |
| `cloud_masking.py` | SCL-layer masking for cloud, shadow, and defective pixels |
| `compositing.py` | Median bi-weekly compositing + harmonic gap-filling |

### `src/features/`

| Module | Description |
|---|---|
| `vegetation_indices.py` | NDVI, EVI, Red Edge CIre at each bi-weekly timestep |
| `phenological_params.py` | Harmonic regression model — amplitude, phase, greenup rate |
| `terrain_derivatives.py` | SRTM elevation, slope, aspect, TWI, TRI |

### `src/models/`

| Module | Description |
|---|---|
| `random_forest.py` | RF classifier (500 trees, grid-search hyperopt, Gini importance) |
| `xgboost_classifier.py` | XGBoost (Bayesian search, early stopping, class-weight adjustment) |
| `cnn_lstm.py` | Hybrid CNN-LSTM (TensorFlow 2.x, Adam, dropout, batch norm) |

### `src/evaluation/`

| Module | Description |
|---|---|
| `accuracy_assessment.py` | Area-adjusted confusion matrices (Olofsson et al. 2014 protocol) |
| `spatial_cv.py` | Spatial block cross-validation (50×50 km blocks, LOO-SCV) |

---

## 📈 Results & Figures

### Figure 1 — Spectral-Phenological Signature of Opium Poppy

```
NDVI
1.0 │                    ╭───╮
0.8 │               ╭───╯   ╰──╮         ← Poppy (peak: Feb–Mar)
0.6 │          ╭───╯            ╰╮
0.4 │     ╭───╯                  ╰╮
0.2 │╭───╯                        ╰───╮
0.0 └──────────────────────────────────────────────────►
    Oct    Nov    Dec    Jan    Feb    Mar    Apr    May
    Sow   Germ   Veg    Veg   Flower  Late  Harvest Fallow

    ─── Opium Poppy   - - - Wheat   ···· Bare/Fallow
```

### Figure 2 — Model Accuracy Comparison

```
Overall Accuracy (%)
    CNN-LSTM  ████████████████████████████████████████████  91.4%
    XGBoost   ████████████████████████████████████████      89.7%
    Rand.For  ████████████████████████████████████          87.3%
              80%                  85%                  90%   95%

Kappa Coefficient
    CNN-LSTM  ████████████████████████████████████████████  0.88
    XGBoost   ████████████████████████████████████████      0.86
    Rand.For  ████████████████████████████████████          0.83
              0.75                 0.82                  0.89  0.95
```

### Figure 3 — Interannual Cultivation Dynamics (Afghanistan)

```
Cultivation Area (×1000 ha)
320 │                         ╭── 313,480
300 │                    ╭───╯
280 │               ╭───╯
260 │          ╭───╯                    ╭── 259,560
240 │     ╭───╯ 225,530           ╭───╯
220 │╭───╯                   ╭───╯
    └──────────────────────────────────────────►
      S1 (2022-23)    S2 (2023-24)    S3 (2024-25)

    █ Satellite estimate   □ UNODC benchmark
    Interannual change: +27.1% → −18.5%
```

### Figure 4 — Risk Surface Tier Distribution

```
Risk Level        Area (km²)    % of Study Area
─────────────────────────────────────────────────
🔴 Very High        4,200           0.6%
🟠 High            14,200           2.1%    ← 18,400 km² unmonitored
🟡 Moderate        48,600           7.1%
🟢 Low            613,000          90.2%
```

> **All output maps (GeoTIFF format), trained model weights and SHAP feature importance plots are available in the `/results/` directory and via the Zenodo data archive linked below.**

---

## 📂 Data Availability

| Dataset | Source | Access |
|---|---|---|
| Sentinel-2 L2A imagery | Copernicus Open Access Hub | [scihub.copernicus.eu](https://scihub.copernicus.eu) - **Free** |
| Google Earth Engine processing | Google / ESA | [earthengine.google.com](https://earthengine.google.com) - **Free** |
| UNODC Opium Survey data | UNODC Data Portal | [dataunodc.un.org](https://dataunodc.un.org) - **Free** |
| SRTM Digital Elevation Model | NASA / USGS | via GEE - **Free** |
| Training sample labels | This study | `/data/sample/` - **Included** |
| Trained model weights | This study | [Zenodo Archive](#) - **Free** |
| Output cultivation maps | This study | [Zenodo Archive](#) - **Free** |

---

## 📖 Citation

If you use this code or data in your research, please cite:

```bibtex
@article{khan2025opium,
  title     = {Geospatial Machine Learning for Opium Poppy Cultivation Monitoring
               in Pakistan and Afghanistan: A Sentinel-2 Multi-Temporal Analysis},
  author    = {Khan, Fahad Hameed},
  journal   = {XXXX},
  publisher = {XXXX},
  year      = {2026},
  volume    = {XX},
  number    = {X},
  pages     = {XXXX},
  doi       = {10.3390/.XXXXXXXX},
  issn      = {2073-445X},
  url       = {https://www.mdpi.com/journal/land}
}
```

---

## 👤 Author

<div align="center">

**Fahad Hameed Khan**
*Independant Researcher*
 Karachi, Pakistan

[![Email](https://img.shields.io/badge/Email-Contact-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:drfahadhameedkhan@gmail.com)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0009--2087--0242-A6CE39?style=for-the-badge&logo=orcid&logoColor=white)](https://orcid.org/0009-0009-2087-0242)
[![ResearchGate](https://img.shields.io/badge/ResearchGate-Profile-00CCBB?style=for-the-badge&logo=researchgate&logoColor=white)](https://www.researchgate.net/profile/Fahad-Khan-119?ev=hdr_xprf)
[![Google Scholar](https://img.shields.io/badge/Google%20Scholar-Profile-4285F4?style=for-the-badge&logo=google-scholar&logoColor=white)](https://scholar.google.com/citations?user=fbh7R64AAAAJ&hl=en)


</div>

**Research Focus:** Geospatial machine learning · Counter-narcotics monitoring · Transnational drug trafficking networks · Computational criminology · Remote sensing for land governance

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

All code and trained models are freely available for academic and non-commercial use. For commercial licensing, please contact the author.

---

## 🙏 Acknowledgements

- **UNODC South Asia Regional Office** for making Annual Opium Survey data publicly accessible
- **European Space Agency** for the free provision of Sentinel-2 imagery through the Copernicus Open Data programme
- **Google Earth Engine** as the primary data processing and cloud computation platform
- **NASA / USGS** for SRTM digital elevation model data

---

<div align="center">

*"Satellite monitoring systems performing at bi-weekly temporal resolution can detect cultivation expansion as it occurs — enabling more timely policy response than annual ground survey programmes."*

**— Khan (2025)**

---

⭐ **If this repository helped your research, please consider starring it!** ⭐

</div>

