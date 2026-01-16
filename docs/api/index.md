# API Documentation

## Overview

The WLAN Indoor Localization package provides a modular, type-safe API for Wi-Fi fingerprint-based indoor positioning.

## Quick Links

- [Data Module](#data-module)
- [Models Module](#models-module)
- [Evaluation Module](#evaluation-module)
- [CLI Commands](#cli-commands)

---

## Data Module

### `wlan_localization.data.loader`

**load_ujiindoorloc()**
```python
def load_ujiindoorloc(
    data_dir: str = "data/raw",
    test_size: float = 0.1,
    random_state: int = 42,
    remove_invalid: bool = True,
    min_aps_detected: int = 1,
    remove_zero_var: bool = True
) -> Tuple[NDArray, DataFrame, NDArray, DataFrame]
```

Load and prepare UJIIndoorLoc dataset.

**Returns**: `(X_train, y_train, X_holdout, y_holdout)`

**Example**:
```python
from wlan_localization.data import load_ujiindoorloc

X_train, y_train, X_test, y_test = load_ujiindoorloc()
print(f"Training samples: {len(X_train)}")
```

---

**UJIIndoorLocDataset**

Class for managing UJIIndoorLoc data loading and preprocessing.

```python
from wlan_localization.data import UJIIndoorLocDataset

dataset = UJIIndoorLocDataset(data_dir="data/raw")
data = dataset.load_raw_data(file_type="training")
X, y = dataset.split_features_labels(data)
```

### `wlan_localization.data.preprocessor`

**DataPreprocessor**

```python
from wlan_localization.data import DataPreprocessor

preprocessor = DataPreprocessor(
    missing_value=100.0,
    apply_box_cox=True,
    n_components=150,
    explained_variance=0.95
)

# Fit on training data
X_transformed = preprocessor.fit_transform(X_train)

# Transform test data
X_test_transformed = preprocessor.transform(X_test)
```

**Key Methods**:
- `fit(X)`: Fit transformations
- `transform(X)`: Apply transformations
- `fit_transform(X)`: Fit and transform in one step
- `get_feature_statistics(X)`: Compute RSSI statistics

---

## Models Module

### `wlan_localization.models.cascade`

**CascadePipeline**

Main interface for the complete cascaded localization system.

```python
from wlan_localization import CascadePipeline

# From configuration file
pipeline = CascadePipeline.from_config('configs/cascade_optimal.yaml')

# Train
pipeline.fit(X_train, y_train)

# Predict
predictions = pipeline.predict(X_test)

# Evaluate
metrics = pipeline.evaluate(X_test, y_test)

# Save/Load
pipeline.save('models/my_cascade')
loaded = CascadePipeline.load('models/my_cascade')
```

**Key Methods**:
- `fit(X, y)`: Train complete pipeline
- `predict(X)`: Predict building, floor, position
- `evaluate(X, y)`: Comprehensive evaluation
- `save(path)`: Serialize pipeline
- `load(path)`: Deserialize pipeline
- `get_stage_predictions(X, y)`: Debug individual stages

### `wlan_localization.models.building_classifier`

**BuildingClassifier**

```python
from wlan_localization.models import BuildingClassifier

clf = BuildingClassifier(
    n_estimators=100,
    random_state=42,
    balancing_strategy="nearmiss"
)

clf.fit(X_train, y_building_train)
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)
metrics = clf.evaluate(X_test, y_building_test)
```

### `wlan_localization.models.floor_classifier`

**FloorClassifier**

```python
from wlan_localization.models import FloorClassifier

clf = FloorClassifier(k=3, metric="manhattan", weights="distance")

# Train per-building models
clf.fit(X_train, y_floor_train, y_building_train)

# Predict (requires building IDs)
predictions = clf.predict(X_test, building_ids)
```

### `wlan_localization.models.position_regressor`

**PositionRegressor**

```python
from wlan_localization.models import PositionRegressor

reg = PositionRegressor(k=3, metric="manhattan", weights="distance")

# Train per-(building,floor) models
reg.fit(X_train, y_position_train, y_building_train, y_floor_train)

# Predict (requires building and floor IDs)
predictions = reg.predict(X_test, building_ids, floor_ids)
```

---

## Evaluation Module

### `wlan_localization.evaluation.metrics`

**compute_positioning_error()**

```python
from wlan_localization.evaluation.metrics import compute_positioning_error

error = compute_positioning_error(
    actual_lat=4864850, actual_lon=-7400,
    actual_building=0, actual_floor=2,
    pred_lat=4864855, pred_lon=-7405,
    pred_building=0, pred_floor=3,
    building_penalty=50.0, floor_penalty=4.0
)
```

**evaluate_cascade_pipeline()**

```python
from wlan_localization.evaluation.metrics import evaluate_cascade_pipeline

metrics = evaluate_cascade_pipeline(
    y_true=y_test,
    y_pred=predictions,
    building_penalty=50.0,
    floor_penalty=4.0
)

print(f"Building accuracy: {metrics['building_accuracy']:.1%}")
print(f"Mean error: {metrics['mean_positioning_error']:.2f}m")
```

**evaluate_per_building_floor()**

```python
from wlan_localization.evaluation.metrics import evaluate_per_building_floor

detailed_metrics = evaluate_per_building_floor(y_true, y_pred)
print(detailed_metrics)
```

### `wlan_localization.evaluation.visualizations`

**Visualization Functions**:

```python
from wlan_localization.evaluation.visualizations import (
    plot_roc_curves,
    plot_confusion_matrix,
    plot_error_distribution,
    plot_spatial_distribution,
    plot_per_location_performance,
    create_evaluation_report
)

# Example: ROC curves
plot_roc_curves(
    y_true=y_building_true,
    y_pred_proba=building_probas,
    class_names=["Building 0", "Building 1", "Building 2"],
    save_path="reports/figures/roc_building.png"
)

# Example: Complete report
create_evaluation_report(
    y_true=y_test,
    y_pred=predictions,
    metrics=metrics,
    output_dir=Path("reports/figures")
)
```

---

## CLI Commands

### wlan-train

Train cascade pipeline from command line.

```bash
wlan-train --config configs/cascade_optimal.yaml \
           --data-dir data/raw \
           --output models/my_model \
           --test-size 0.1 \
           --mlflow
```

**Options**:
- `--config`: Path to YAML configuration file
- `--data-dir`: Directory with training data
- `--output`: Output directory for trained model
- `--test-size`: Holdout set proportion (0.0-1.0)
- `--random-state`: Random seed
- `--mlflow/--no-mlflow`: Enable MLflow tracking
- `--experiment-name`: MLflow experiment name
- `--run-name`: MLflow run name

### wlan-evaluate

Evaluate trained model.

```bash
wlan-evaluate --model models/my_model \
              --data-dir data/raw \
              --output reports/metrics/eval.json \
              --save-predictions reports/predictions.csv \
              --detailed
```

**Options**:
- `--model`: Path to trained model directory
- `--data-dir`: Directory with test data
- `--output`: Output file for metrics (JSON)
- `--save-predictions`: Save predictions to CSV
- `--building-penalty`: Building error penalty (default: 50.0)
- `--floor-penalty`: Floor error penalty (default: 4.0)
- `--detailed/--no-detailed`: Per-location metrics

### wlan-predict

Make predictions on new data.

```bash
wlan-predict --model models/my_model \
             --input data/new_measurements.csv \
             --output predictions.csv \
             --format csv
```

**Options**:
- `--model`: Path to trained model directory
- `--input`: Input CSV with RSSI measurements
- `--output`: Output file for predictions
- `--format`: Output format (csv or json)
- `--batch-size`: Batch size for large files

---

## Type Annotations

All modules use comprehensive type hints for better IDE support:

```python
from typing import Tuple, Optional
from numpy.typing import NDArray
import numpy as np
import pandas as pd

def example_function(
    X: NDArray[np.float64],
    y: Optional[pd.DataFrame] = None
) -> Tuple[NDArray[np.float64], dict]:
    ...
```

---

## Configuration Files

Example configuration (YAML):

```yaml
# configs/my_experiment.yaml
experiment:
  name: "my_experiment"
  seed: 42

preprocessing:
  missing_value: 100.0
  apply_box_cox: true
  n_components: 150

models:
  building_classifier:
    n_estimators: 100
    balancing_strategy: "nearmiss"

  floor_classifier:
    k: 3
    metric: "manhattan"
    weights: "distance"

  position_regressor:
    k: 3
    metric: "manhattan"
    weights: "distance"
```

Load configuration:

```python
pipeline = CascadePipeline.from_config('configs/my_experiment.yaml')
```

---

## Error Handling

All methods raise informative exceptions:

```python
from wlan_localization import CascadePipeline

pipeline = CascadePipeline()

try:
    # This will raise RuntimeError
    predictions = pipeline.predict(X_test)
except RuntimeError as e:
    print(f"Error: {e}")  # "Pipeline not fitted. Call fit() first."
```

---

## Logging

Structured logging with loguru:

```python
from wlan_localization.utils.logger import get_logger

logger = get_logger(__name__)

logger.info("Training started")
logger.debug("Detailed information")
logger.warning("Warning message")
logger.error("Error occurred")
```

Logs are saved to `logs/wlan_localization_*.log` with rotation.

---

## For More Information

- **Source Code**: See `src/wlan_localization/` directory
- **Methodology**: See `docs/methodology.md`
- **Examples**: See `notebooks/` directory
- **Tests**: See `tests/` directory

---

Generated with Python 3.11+ type annotations and comprehensive docstrings.
