# WLAN Indoor Localization using Machine Learning

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-ready machine learning system for Wi-Fi fingerprint-based indoor positioning using a novel cascaded architecture. Achieves 2.6-8.2m positioning accuracy across multiple buildings and floors.

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Methodology](#methodology)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Development](#development)
- [Citation](#citation)
- [License](#license)

## Overview

Indoor localization remains a challenging problem due to GPS signal loss in enclosed environments. This project implements a sophisticated machine learning solution for Wi-Fi fingerprint-based positioning that:

- **Handles complex multi-building, multi-floor environments** with a cascaded classification-regression approach
- **Achieves competitive accuracy** (2.6-8.2m RMSE) on the benchmark UJIIndoorLoc dataset
- **Implements modern ML engineering practices** with modular code, comprehensive testing, and experiment tracking
- **Addresses real-world challenges** including 96% data sparsity, class imbalance, and high dimensionality (520 Wi-Fi access points)

### Business Value

Indoor positioning enables critical applications including:
- **Navigation**: Indoor navigation in airports, malls, hospitals
- **Asset Tracking**: Real-time equipment and inventory tracking in warehouses
- **Emergency Response**: First responder location tracking in buildings
- **Analytics**: Customer behavior analysis in retail environments

## Key Results

### Positioning Accuracy

| Metric | Value |
|--------|-------|
| **Building Classification** | 99.4% accuracy |
| **Floor Classification** | 94-98% accuracy (per-building models) |
| **Best Position Accuracy** | 2.65m RMSE (Building 1, Floor 2) |
| **Average Position Accuracy** | 3-8m RMSE (varies by location) |
| **Global Positioning Error** | 5.28m (including building/floor penalties) |

### Performance by Location

```
Building 0:
├── Floor 0: 3.72m RMSE
├── Floor 1: 3.18m RMSE
├── Floor 2: 3.71m RMSE
└── Floor 3: 3.32m RMSE

Building 1:
├── Floor 0: 4.07m RMSE
├── Floor 1: 5.28m RMSE
├── Floor 2: 2.65m RMSE (best)
└── Floor 3: 4.79m RMSE

Building 2:
├── Floor 0: 4.06m RMSE
├── Floor 1: 4.06m RMSE
├── Floor 2: 3.76m RMSE
├── Floor 3: 2.77m RMSE
└── Floor 4: 6.76m RMSE
```

## Methodology

### Cascaded ML Architecture

Our approach uses a three-stage cascade pipeline that dramatically outperforms global models:

```
┌─────────────────────────────────────────────────┐
│  Stage 1: Building Classification               │
│  Algorithm: Random Forest (100 trees)           │
│  Class Balancing: NearMiss undersampling        │
│  Accuracy: 99.4%                                │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│  Stage 2: Per-Building Floor Classification     │
│  Algorithm: Weighted KNN (k=3, manhattan)       │
│  Separate model per building                    │
│  Accuracy: 94-98% per building                  │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│  Stage 3: Per-Building-Floor Position Regression│
│  Algorithm: Weighted KNN (k=3, distance-weighted)│
│  13 separate models (per building-floor combo)  │
│  Output: Latitude, Longitude                    │
└─────────────────────────────────────────────────┘
```

### Why Cascade Outperforms Global Models

The cascade approach provides:
1. **Specialized models** tuned to each building's unique RF characteristics
2. **Better handling of spatial variation** across different floors
3. **Interpretable predictions** with confidence at each stage
4. **Custom error metric** that penalizes building/floor misclassification:

```
positioning_error = euclidean_distance + 50 × building_error + 4 × floor_error
```

### Data Processing Pipeline

1. **Missing Value Handling**: RSSI signals have 96% sparsity (out-of-range = 100 dBm)
2. **Dimensionality Reduction**: PCA reduces 520 AP features → 150 components (95% variance)
3. **Box-Cox Transformation**: Addresses right-skewed RSSI distributions
4. **Feature Engineering**: Statistical features (skewness, kurtosis) for signal quality

### Model Selection Process

Comprehensive comparison of algorithms:
- **Linear Models**: Ridge, Lasso (baseline: ~25m RMSE)
- **Polynomial Regression**: Quadratic/cubic features
- **K-Nearest Neighbors**: ✅ Selected (5.69m RMSE, best for spatial data)
- **Random Forests**: 6.78m RMSE
- **Extra Trees**: 8.89m RMSE
- **XGBoost**: Evaluated but KNN superior for this problem

**KNN won** due to:
- Natural fit for spatial similarity
- No overfitting on high-dimensional sparse data
- Fast inference with distance weighting
- Interpretable neighbor-based predictions

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/wlan_localization
cd wlan_localization

# Create virtual environment (Python 3.11+)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .

# Download UJIIndoorLoc dataset
python scripts/download_data.py

# Train the cascade model
wlan-train --config configs/cascade_optimal.yaml

# Evaluate on test set
wlan-evaluate --model models/cascade_best.pkl --data data/processed/test.csv

# Make predictions
wlan-predict --model models/cascade_best.pkl --rssi "[-90,-85,-92,...]"
```

## Installation

### Requirements

- Python 3.11 or higher
- 4GB RAM minimum (8GB recommended for training)
- 500MB disk space for data and models

### From Source

```bash
# Clone repository
git clone https://github.com/yourusername/wlan_localization
cd wlan_localization

# Install in development mode with all dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Dependencies

Core dependencies:
- `scikit-learn>=1.4.0` - Machine learning algorithms
- `pandas>=2.2.0` - Data manipulation
- `numpy>=1.26.0` - Numerical operations
- `imbalanced-learn>=0.12.0` - Class balancing (SMOTE, NearMiss)
- `mlflow>=2.10.0` - Experiment tracking
- `optuna>=3.5.0` - Hyperparameter optimization

See `pyproject.toml` for complete dependency list.

## Usage

### Python API

```python
from wlan_localization import CascadePipeline
from wlan_localization.data import load_ujiindoorloc
import numpy as np

# Load data
X_train, y_train, X_test, y_test = load_ujiindoorloc()

# Initialize and train cascade pipeline
pipeline = CascadePipeline.from_config('configs/cascade_optimal.yaml')
pipeline.fit(X_train, y_train)

# Make predictions
predictions = pipeline.predict(X_test)
print(f"Building: {predictions['building']}")
print(f"Floor: {predictions['floor']}")
print(f"Position: ({predictions['latitude']}, {predictions['longitude']})")

# Evaluate
metrics = pipeline.evaluate(X_test, y_test)
print(f"Mean positioning error: {metrics['positioning_error']:.2f}m")
print(f"Building accuracy: {metrics['building_accuracy']:.1%}")
print(f"Floor accuracy: {metrics['floor_accuracy']:.1%}")
```

### Command Line Interface

```bash
# Train with custom configuration
wlan-train \
    --config configs/my_experiment.yaml \
    --data data/processed/train.csv \
    --output models/my_model.pkl

# Evaluate with detailed metrics
wlan-evaluate \
    --model models/cascade_best.pkl \
    --data data/processed/test.csv \
    --output reports/metrics/evaluation.json \
    --plot-roc \
    --plot-confusion

# Predict single sample
wlan-predict \
    --model models/cascade_best.pkl \
    --rssi-file data/samples/single_measurement.csv \
    --format json
```

### Jupyter Notebooks

Explore the analysis notebooks:

1. **01_exploratory_data_analysis.ipynb** - Dataset exploration and visualization
2. **02_data_preprocessing.ipynb** - Feature engineering and PCA
3. **03_model_development.ipynb** - Model comparison and selection
4. **04_cascade_pipeline.ipynb** - Building the cascade system
5. **05_results_analysis.ipynb** - Performance evaluation and visualization

```bash
# Launch Jupyter
jupyter notebook notebooks/
```

## Dataset

### UJIIndoorLoc

The [UJIIndoorLoc dataset](https://archive.ics.uci.edu/ml/datasets/UJIIndoorLoc) is a benchmark for indoor localization research from the Universitat Jaume I.

**Dataset Statistics:**
- **Training samples**: 19,937
- **Test samples**: 1,111
- **Wi-Fi Access Points**: 520
- **Buildings**: 3
- **Floors**: 5 (0-4, not all in every building)
- **Coverage area**: ~108,703 m²
- **Devices**: 25 different smartphones

**Features:**
- **RSSI values**: Signal strength in dBm (range: -104 to 0, missing = 100)
- **Location labels**: Building ID, Floor ID, Latitude, Longitude
- **Metadata**: User ID, Phone ID, Timestamp, Space ID, Relative Position

**Data Characteristics:**
- **High sparsity**: 96% of RSSI values are out-of-range
- **Class imbalance**: Building 2 has 43% of samples, Buildings 0 & 1 have ~28% each
- **Spatial heterogeneity**: Different RF characteristics per building/floor

### Download Instructions

```bash
# Automated download (recommended)
python scripts/download_data.py

# Manual download
# 1. Visit: https://archive.ics.uci.edu/ml/datasets/UJIIndoorLoc
# 2. Download trainingData.csv and validationData.csv
# 3. Place in data/raw/
```

## Project Structure

```
wlan_localization/
├── configs/                      # Experiment configurations
│   ├── default.yaml             # Default hyperparameters
│   └── cascade_optimal.yaml     # Best performing configuration
├── data/                        # Data directory (not in git)
│   ├── raw/                     # Original UCI dataset
│   ├── processed/               # Preprocessed features
│   └── splits/                  # Train/val/test splits
├── docs/                        # Extended documentation
│   ├── methodology.md           # Detailed ML methodology
│   ├── experiments.md           # Experiment tracking
│   └── api/                     # API documentation
├── models/                      # Trained models (not in git)
│   ├── building_classifier/
│   ├── floor_classifiers/
│   └── position_regressors/
├── notebooks/                   # Jupyter notebooks
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_development.ipynb
│   ├── 04_cascade_pipeline.ipynb
│   └── 05_results_analysis.ipynb
├── reports/                     # Generated reports
│   ├── figures/                 # Visualizations
│   └── metrics/                 # Evaluation metrics
├── scripts/                     # Utility scripts
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation script
│   ├── predict.py              # Inference script
│   └── download_data.py        # Data download
├── src/wlan_localization/      # Source code
│   ├── __init__.py
│   ├── cli/                    # Command-line interfaces
│   ├── data/                   # Data loading and preprocessing
│   │   ├── loader.py
│   │   ├── preprocessor.py
│   │   └── validator.py
│   ├── evaluation/             # Metrics and visualization
│   │   ├── metrics.py
│   │   └── visualizations.py
│   ├── features/               # Feature engineering
│   │   ├── engineering.py
│   │   └── selection.py
│   ├── models/                 # ML models
│   │   ├── base.py
│   │   ├── building_classifier.py
│   │   ├── cascade.py
│   │   ├── floor_classifier.py
│   │   └── position_regressor.py
│   └── utils/                  # Utilities
│       ├── config.py
│       ├── logger.py
│       └── paths.py
├── tests/                      # Test suite
│   ├── test_data/
│   ├── test_features/
│   ├── test_models/
│   └── conftest.py
├── .github/workflows/          # CI/CD pipelines
│   └── ci.yml
├── .gitignore
├── .pre-commit-config.yaml     # Pre-commit hooks
├── LICENSE                     # MIT License
├── pyproject.toml             # Project configuration
└── README.md                  # This file
```

## Technical Details

### Machine Learning Pipeline

**1. Data Preprocessing**
- Missing value imputation (100 dBm → out-of-range handling)
- Box-Cox transformation for normality
- PCA dimensionality reduction (520 → 150 features)
- Standard scaling for KNN distance metrics

**2. Building Classification**
- Algorithm: Random Forest (100 estimators)
- Class balancing: NearMiss-2 undersampling
- Cross-validation: 5-fold stratified
- Metric: Accuracy, ROC-AUC per class

**3. Floor Classification (Per-Building)**
- Algorithm: Weighted KNN (k=3, manhattan distance)
- 3 separate models (one per building)
- Improves accuracy from ~85% (global) to 94-98% (local)
- Handles different floor counts per building

**4. Position Regression (Per-Building-Floor)**
- Algorithm: Weighted KNN (k=3, distance-weighted)
- 13 separate models (one per building-floor combination)
- Multivariate output: (latitude, longitude)
- Improves RMSE from 5.69m (global) to 2.65-8.17m (local)

**5. Model Selection**
- Nested cross-validation (outer: 5-fold, inner: 2-fold)
- Grid search hyperparameter tuning
- Evaluation: RMSE, MAE, R² for regression; Accuracy, F1 for classification

### Experiments and Reproducibility

All experiments tracked with MLflow:

```bash
# View experiment results
mlflow ui

# Compare runs
mlflow experiments list
```

Configuration-driven experiments:
```yaml
# configs/cascade_optimal.yaml
models:
  building_classifier:
    type: "RandomForest"
    n_estimators: 100
    class_balancing: "NearMiss"

  floor_classifier:
    type: "KNN"
    k: 3
    metric: "manhattan"
    weights: "distance"

  position_regressor:
    type: "WeightedKNN"
    k: 3
```

### Performance Optimization

- **Training time**: ~15 minutes on CPU (MacBook Pro M1)
- **Inference time**: <10ms per prediction
- **Model size**: ~500MB (13 KNN models + training data)
- **Memory footprint**: ~2GB during training

## Development

### Setup Development Environment

```bash
# Clone and install with dev dependencies
git clone https://github.com/yourusername/wlan_localization
cd wlan_localization
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=wlan_localization --cov-report=html

# Run specific test file
pytest tests/test_models/test_cascade.py

# Run tests in parallel
pytest -n auto
```

### Code Quality

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint
flake8 src/ tests/

# Type check
mypy src/
```

### Pre-commit Hooks

Automatically run on every commit:
- `black` - Code formatting
- `isort` - Import sorting
- `flake8` - Linting
- `mypy` - Type checking

```bash
# Run manually
pre-commit run --all-files
```

### Continuous Integration

GitHub Actions automatically:
- Runs test suite on Python 3.11 & 3.12
- Checks code formatting and linting
- Runs type checking
- Generates coverage reports
- Deploys documentation

## Citation

If you use the UJIIndoorLoc dataset, please cite:

```bibtex
@inproceedings{torres2014ujiindoorloc,
  title={UJIIndoorLoc: A new multi-building and multi-floor database for WLAN fingerprint-based indoor localization problems},
  author={Torres-Sospedra, Joaqu{\'\i}n and Montoliu, Ra{\'u}l and Mart{\'\i}nez-Us{\'o}, Adolfo and Avariento, Joan P and Arnau, Tom{\'a}s J and Benedito-Bordonau, Mauri and Huerta, Joaqu{\'\i}n},
  booktitle={2014 international conference on indoor positioning and indoor navigation (IPIN)},
  pages={261--270},
  year={2014},
  organization={IEEE}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **UJIIndoorLoc Dataset**: Universitat Jaume I for providing the benchmark dataset
- **Scikit-learn**: Machine learning library
- **MLflow**: Experiment tracking platform
- **Community**: Open-source ML community for tools and inspiration

---

**For questions or collaboration:** [Open an issue](https://github.com/yourusername/wlan_localization/issues)

**Star this repo** if you found it useful!
