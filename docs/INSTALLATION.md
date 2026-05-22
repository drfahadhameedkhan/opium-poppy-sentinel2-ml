# Installation Guide

## Prerequisites

- **Python:** Version 3.9 or higher
- **RAM:** ≥16 GB recommended (for CNN-LSTM training)
- **GPU:** Optional but recommended for deep learning
- **Google Earth Engine:** Free account at [earthengine.google.com](https://earthengine.google.com)

## Step 1: Clone Repository

```bash
git clone https://github.com/drfahadhameedkhan/opium-poppy-sentinel2-ml.git
cd opium-poppy-sentinel2-ml
```

## Step 2: Create Virtual Environment

### Option A: Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate poppy-monitor
```

### Option B: pip + venv

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Step 3: Install Development Tools (Optional)

For contributing or development:

```bash
pip install -r requirements-dev.txt
```

## Step 4: Authenticate Google Earth Engine

```bash
earthengine authenticate
```

## Step 5: Verify Installation

```bash
python -c "import src; print('✅ Installation successful!')"
```

## Troubleshooting

### Issue: Module import errors

**Solution:** Ensure you're in the correct directory:
```bash
cd /path/to/opium-poppy-sentinel2-ml
python -c "import src"
```

### Issue: TensorFlow GPU not detected

**Solution:**
```bash
pip install tensorflow[and-cuda]
```

### Issue: Out of memory during training

**Solution:** Reduce batch size in `configs/model_config.yaml`
