# Jupyter Notebooks

## Overview

This directory contains the original research notebooks that led to the development of the WLAN Indoor Localization cascade pipeline.

## Notebooks

### 1. UJIIndoorLoc-preprocess.ipynb (598KB)

**Purpose**: Data loading and preprocessing

**Contents**:
- Loading UJIIndoorLoc dataset
- Exploratory Data Analysis (EDA)
- Missing value analysis (96% sparsity)
- Box-Cox transformation for normality
- PCA dimensionality reduction (520 → 150 components)
- Statistical analysis (skewness, kurtosis)
- Data export for modeling

**Key Outputs**:
- `data/X_pca_crossval.csv` - PCA-transformed training features
- `data/X_pca_holdout.csv` - PCA-transformed holdout features
- `data/y_crossval.csv` - Training labels
- `data/y_holdout.csv` - Holdout labels

**Run Time**: ~5-10 minutes

---

### 2. UJIIndoorLoc-machine-learning.ipynb (933KB)

**Purpose**: Model development and comparison

**Contents**:
- Response variable EDA (building, floor, position distributions)
- Problem formulation with custom error metric
- Nested cross-validation framework
- Model comparison:
  - Linear models (Ridge, Lasso)
  - Polynomial regression
  - K-Nearest Neighbors ✅ (selected)
  - Random Forests
  - Extra Trees
- Hyperparameter tuning
- Per-building-floor model training
- Final cascade pipeline construction
- Results visualization

**Key Findings**:
- KNN outperforms other methods (5.69m global RMSE)
- Per-building-floor models achieve 2.6-8.2m RMSE
- Manhattan distance superior to Euclidean for sparse data
- k=3 optimal for both floor and position models

**Run Time**: ~30-60 minutes (nested cross-validation)

---

### 3. UJIIndoorLoc-response-classification.ipynb (273KB)

**Purpose**: Building and floor classification

**Contents**:
- Building classification with Random Forest
- Class imbalance handling (SMOTE, NearMiss)
- ROC curve analysis
- Floor classification (global and per-building)
- Confusion matrix analysis
- Per-building floor models comparison

**Key Results**:
- Building classification: 99.4% accuracy
- Floor classification (per-building): 94-98% accuracy
- NearMiss-2 undersampling most effective

**Run Time**: ~10-15 minutes

---

### 4. Kernel_density.ipynb (171KB)

**Purpose**: Experimental missing value imputation

**Contents**:
- Kernel Density Estimation (KDE) experiments
- Alternative missing value handling strategies
- Comparison to indicator approach

**Conclusion**: Indicator strategy (keeping 100 as-is) performed best

**Run Time**: ~5 minutes

---

## Usage

### Prerequisites

```bash
# Install dependencies
pip install -e ".[dev]"

# Download data
python scripts/download_data.py
```

### Running Notebooks

```bash
# Launch Jupyter
jupyter notebook notebooks/

# Or use JupyterLab
jupyter lab notebooks/
```

### Recommended Order

1. **UJIIndoorLoc-preprocess.ipynb** - Understand data and preprocessing
2. **UJIIndoorLoc-response-classification.ipynb** - Classification stages
3. **UJIIndoorLoc-machine-learning.ipynb** - Complete pipeline
4. **Kernel_density.ipynb** - Optional (experimental)

---

## Modernization Notes

These notebooks represent the original 2017 research. The modernized production code is in:
- `src/wlan_localization/` - Modular, tested, type-safe implementation
- `tests/` - Comprehensive test suite
- `configs/` - Configuration-driven experiments

**Production Code Advantages**:
- ✅ Modular and reusable
- ✅ Type hints and docstrings
- ✅ Comprehensive tests
- ✅ CI/CD integration
- ✅ CLI interface
- ✅ MLflow tracking

**Notebooks Advantages**:
- ✅ Exploratory analysis
- ✅ Visualization-rich
- ✅ Narrative explanation
- ✅ Experimentation

---

## Converting Notebooks to Production

**From Notebook**:
```python
# Notebook cell
preprocessor = DataPreprocessor(n_components=150)
X_transformed = preprocessor.fit_transform(X_train)
```

**To Production**:
```python
# src/wlan_localization/data/preprocessor.py
from wlan_localization.data import DataPreprocessor

preprocessor = DataPreprocessor(n_components=150)
X_transformed = preprocessor.fit_transform(X_train)
```

**With CLI**:
```bash
# configs/my_experiment.yaml
preprocessing:
  n_components: 150

# Run
wlan-train --config configs/my_experiment.yaml
```

---

## Cleaning Notebooks

Clear outputs before committing:

```bash
# Install nbstripout
pip install nbstripout

# Strip outputs
nbstripout notebooks/*.ipynb

# Or configure pre-commit hook (already set up)
pre-commit install
```

---

## Known Issues

**Original Notebooks (2017)**:
- Some deprecated sklearn API calls (e.g., `Imputer` → `SimpleImputer`)
- Large pickle files should be regenerated
- Magic numbers hardcoded (now in configs)
- No systematic experiment tracking (now using MLflow)

**Resolution**: Use modernized `src/` code for production, notebooks for reference

---

## Citation

If you use these notebooks or methodologies, please cite:

```bibtex
@software{naribole2026wlan,
  author = {Naribole, Sharan},
  title = {WLAN Indoor Localization using Machine Learning},
  year = {2026},
  url = {https://github.com/yourusername/wlan_localization}
}
```

Original dataset:
```bibtex
@inproceedings{torres2014ujiindoorloc,
  title={UJIIndoorLoc: A new multi-building and multi-floor database for WLAN fingerprint-based indoor localization problems},
  author={Torres-Sospedra, Joaqu{\'\i}n and others},
  booktitle={IPIN 2014},
  year={2014}
}
```

---

## Questions?

- **Documentation**: See `docs/methodology.md` for detailed explanation
- **API Reference**: See `docs/api/index.md`
- **Issues**: Open an issue on GitHub
- **Contributing**: See `CONTRIBUTING.md`
