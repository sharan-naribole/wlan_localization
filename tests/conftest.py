"""Pytest configuration and fixtures."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_rssi_data():
    """Generate synthetic RSSI data for testing.

    Returns:
        Tuple of (X, y) where:
            X: RSSI feature array (100 samples, 520 APs)
            y: DataFrame with LATITUDE, LONGITUDE, BUILDINGID, FLOOR
    """
    np.random.seed(42)

    n_samples = 100
    n_aps = 520
    missing_value = 100.0

    # Generate RSSI values (mostly out-of-range)
    X = np.full((n_samples, n_aps), missing_value)

    # Make some APs in range for each sample
    for i in range(n_samples):
        n_in_range = np.random.randint(10, 30)
        in_range_indices = np.random.choice(n_aps, n_in_range, replace=False)
        X[i, in_range_indices] = np.random.uniform(-95, -40, n_in_range)

    # Generate labels
    y = pd.DataFrame({
        "LATITUDE": np.random.uniform(4864746, 4865017, n_samples),
        "LONGITUDE": np.random.uniform(-7691, -7300, n_samples),
        "BUILDINGID": np.random.choice([0, 1, 2], n_samples),
        "FLOOR": np.random.choice([0, 1, 2, 3, 4], n_samples),
    })

    return X, y


@pytest.fixture
def small_rssi_data():
    """Generate small RSSI dataset for quick tests.

    Returns:
        Tuple of (X, y) with 20 samples
    """
    np.random.seed(42)

    n_samples = 20
    n_aps = 50
    missing_value = 100.0

    X = np.full((n_samples, n_aps), missing_value)

    for i in range(n_samples):
        n_in_range = np.random.randint(5, 15)
        in_range_indices = np.random.choice(n_aps, n_in_range, replace=False)
        X[i, in_range_indices] = np.random.uniform(-90, -50, n_in_range)

    y = pd.DataFrame({
        "LATITUDE": np.random.uniform(4864746, 4865017, n_samples),
        "LONGITUDE": np.random.uniform(-7691, -7300, n_samples),
        "BUILDINGID": np.random.choice([0, 1, 2], n_samples),
        "FLOOR": np.random.choice([0, 1, 2], n_samples),
    })

    return X, y


@pytest.fixture
def balanced_classification_data():
    """Generate balanced classification dataset.

    Returns:
        Tuple of (X, y) with equal samples per class
    """
    np.random.seed(42)

    n_per_class = 30
    n_features = 50

    # Building 0
    X_b0 = np.random.randn(n_per_class, n_features) - 1
    y_b0 = np.zeros(n_per_class, dtype=int)

    # Building 1
    X_b1 = np.random.randn(n_per_class, n_features)
    y_b1 = np.ones(n_per_class, dtype=int)

    # Building 2
    X_b2 = np.random.randn(n_per_class, n_features) + 1
    y_b2 = np.full(n_per_class, 2, dtype=int)

    X = np.vstack([X_b0, X_b1, X_b2])
    y = np.hstack([y_b0, y_b1, y_b2])

    return X, y


@pytest.fixture
def sample_predictions():
    """Generate sample predictions for testing metrics.

    Returns:
        Tuple of (y_true, y_pred) DataFrames
    """
    np.random.seed(42)

    n_samples = 50

    y_true = pd.DataFrame({
        "LATITUDE": np.random.uniform(4864746, 4865017, n_samples),
        "LONGITUDE": np.random.uniform(-7691, -7300, n_samples),
        "BUILDINGID": np.random.choice([0, 1, 2], n_samples),
        "FLOOR": np.random.choice([0, 1, 2, 3], n_samples),
    })

    # Generate predictions with some error
    y_pred = y_true.copy()
    y_pred["LATITUDE"] += np.random.randn(n_samples) * 5  # 5m std error
    y_pred["LONGITUDE"] += np.random.randn(n_samples) * 5

    # Add some classification errors
    building_error_mask = np.random.rand(n_samples) < 0.05  # 5% error rate
    n_building_errors = building_error_mask.sum()
    y_pred.loc[building_error_mask, "BUILDINGID"] = np.random.choice([0, 1, 2], n_building_errors)

    floor_error_mask = np.random.rand(n_samples) < 0.1  # 10% error rate
    y_pred.loc[floor_error_mask, "FLOOR"] = np.random.choice([0, 1, 2, 3], floor_error_mask.sum())

    return y_true, y_pred


@pytest.fixture
def temp_model_dir(tmp_path):
    """Create temporary directory for model saving/loading.

    Args:
        tmp_path: Pytest's temporary directory fixture

    Returns:
        Path to temporary model directory
    """
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir
