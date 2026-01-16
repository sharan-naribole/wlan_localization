"""Custom evaluation metrics for indoor localization."""

from typing import Dict, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, mean_squared_error

from wlan_localization.utils.logger import get_logger

logger = get_logger(__name__)


def compute_positioning_error(
    actual_lat: float,
    actual_lon: float,
    actual_building: int,
    actual_floor: int,
    pred_lat: float,
    pred_lon: float,
    pred_building: int,
    pred_floor: int,
    building_penalty: float = 50.0,
    floor_penalty: float = 4.0
) -> float:
    """Compute positioning error with building/floor misclassification penalties.

    This metric was introduced in the 2015 EvAAL-ETRI competition.
    It penalizes both Euclidean distance error and classification failures.

    Args:
        actual_lat: Actual latitude
        actual_lon: Actual longitude
        actual_building: Actual building ID
        actual_floor: Actual floor ID
        pred_lat: Predicted latitude
        pred_lon: Predicted longitude
        pred_building: Predicted building ID
        pred_floor: Predicted floor ID
        building_penalty: Penalty for building misclassification (default: 50m)
        floor_penalty: Penalty for floor misclassification (default: 4m)

    Returns:
        Total positioning error in meters

    Example:
        >>> error = compute_positioning_error(
        ...     actual_lat=4864850, actual_lon=-7400,
        ...     actual_building=0, actual_floor=2,
        ...     pred_lat=4864855, pred_lon=-7405,
        ...     pred_building=0, pred_floor=3
        ...  )
        >>> print(f"Error: {error:.2f}m")  # Euclidean + floor_penalty
    """
    # Euclidean distance error
    euclidean_dist = np.sqrt((pred_lat - actual_lat) ** 2 + (pred_lon - actual_lon) ** 2)

    # Classification penalties
    building_error = building_penalty if pred_building != actual_building else 0.0
    floor_error = floor_penalty if pred_floor != actual_floor else 0.0

    total_error = euclidean_dist + building_error + floor_error

    return total_error


def compute_rmse(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64]
) -> float:
    """Compute Root Mean Squared Error.

    Args:
        y_true: True values (can be 1D or 2D for multivariate)
        y_pred: Predicted values

    Returns:
        RMSE value
    """
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)


def compute_mae(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64]
) -> float:
    """Compute Mean Absolute Error.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MAE value
    """
    return np.abs(y_true - y_pred).mean()


def evaluate_cascade_pipeline(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    building_penalty: float = 50.0,
    floor_penalty: float = 4.0
) -> Dict[str, float]:
    """Evaluate complete cascade pipeline predictions.

    Computes comprehensive metrics including:
    - Positioning error (with penalties)
    - Building classification accuracy
    - Floor classification accuracy
    - Position RMSE (Euclidean distance only)
    - Per-building metrics

    Args:
        y_true: DataFrame with columns ['LATITUDE', 'LONGITUDE', 'BUILDINGID', 'FLOOR']
        y_pred: DataFrame with same columns as predictions
        building_penalty: Penalty for building misclassification
        floor_penalty: Penalty for floor misclassification

    Returns:
        Dictionary of evaluation metrics

    Example:
        >>> metrics = evaluate_cascade_pipeline(y_test, y_pred)
        >>> print(f"Building accuracy: {metrics['building_accuracy']:.1%}")
        >>> print(f"Mean positioning error: {metrics['mean_positioning_error']:.2f}m")
    """
    n_samples = len(y_true)

    # Building classification accuracy
    building_accuracy = accuracy_score(
        y_true["BUILDINGID"],
        y_pred["BUILDINGID"]
    )

    # Floor classification accuracy
    floor_accuracy = accuracy_score(
        y_true["FLOOR"],
        y_pred["FLOOR"]
    )

    # Compute positioning errors for each sample
    positioning_errors = []
    euclidean_distances = []

    for idx in range(n_samples):
        error = compute_positioning_error(
            actual_lat=y_true.iloc[idx]["LATITUDE"],
            actual_lon=y_true.iloc[idx]["LONGITUDE"],
            actual_building=y_true.iloc[idx]["BUILDINGID"],
            actual_floor=y_true.iloc[idx]["FLOOR"],
            pred_lat=y_pred.iloc[idx]["LATITUDE"],
            pred_lon=y_pred.iloc[idx]["LONGITUDE"],
            pred_building=y_pred.iloc[idx]["BUILDINGID"],
            pred_floor=y_pred.iloc[idx]["FLOOR"],
            building_penalty=building_penalty,
            floor_penalty=floor_penalty
        )
        positioning_errors.append(error)

        # Also compute pure Euclidean distance
        eucl_dist = np.sqrt(
            (y_pred.iloc[idx]["LATITUDE"] - y_true.iloc[idx]["LATITUDE"]) ** 2 +
            (y_pred.iloc[idx]["LONGITUDE"] - y_true.iloc[idx]["LONGITUDE"]) ** 2
        )
        euclidean_distances.append(eucl_dist)

    # Aggregate metrics
    metrics = {
        "building_accuracy": building_accuracy,
        "floor_accuracy": floor_accuracy,
        "mean_positioning_error": np.mean(positioning_errors),
        "std_positioning_error": np.std(positioning_errors),
        "median_positioning_error": np.median(positioning_errors),
        "mean_euclidean_distance": np.mean(euclidean_distances),
        "std_euclidean_distance": np.std(euclidean_distances),
        "median_euclidean_distance": np.median(euclidean_distances),
        "max_positioning_error": np.max(positioning_errors),
        "building_penalty_applied": building_penalty,
        "floor_penalty_applied": floor_penalty,
    }

    # Per-building metrics
    for building in y_true["BUILDINGID"].unique():
        mask = y_true["BUILDINGID"] == building
        if mask.sum() > 0:
            building_floor_acc = accuracy_score(
                y_true[mask]["FLOOR"],
                y_pred[mask]["FLOOR"]
            )
            metrics[f"building_{building}_floor_accuracy"] = building_floor_acc

            # Position RMSE for this building
            building_eucl = [euclidean_distances[i] for i, m in enumerate(mask) if m]
            metrics[f"building_{building}_mean_distance"] = np.mean(building_eucl)

    logger.info("Evaluation metrics computed")
    logger.info(f"Building accuracy: {metrics['building_accuracy']:.1%}")
    logger.info(f"Floor accuracy: {metrics['floor_accuracy']:.1%}")
    logger.info(f"Mean positioning error: {metrics['mean_positioning_error']:.2f}m")
    logger.info(f"Mean Euclidean distance: {metrics['mean_euclidean_distance']:.2f}m")

    return metrics


def evaluate_per_building_floor(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame
) -> pd.DataFrame:
    """Compute metrics broken down by building and floor.

    Args:
        y_true: True labels
        y_pred: Predictions

    Returns:
        DataFrame with per-(building,floor) metrics
    """
    results = []

    for building in y_true["BUILDINGID"].unique():
        building_mask = (y_true["BUILDINGID"] == building) & \
                       (y_pred["BUILDINGID"] == building)

        if building_mask.sum() == 0:
            continue

        for floor in y_true[building_mask]["FLOOR"].unique():
            floor_mask = building_mask & (y_true["FLOOR"] == floor)

            if floor_mask.sum() == 0:
                continue

            # Compute RMSE for this building-floor combo
            y_true_subset = y_true[floor_mask][["LATITUDE", "LONGITUDE"]].values
            y_pred_subset = y_pred[floor_mask][["LATITUDE", "LONGITUDE"]].values

            rmse = compute_rmse(y_true_subset, y_pred_subset)
            mae = compute_mae(y_true_subset, y_pred_subset)

            results.append({
                "building": building,
                "floor": floor,
                "n_samples": floor_mask.sum(),
                "rmse": rmse,
                "mae": mae
            })

    results_df = pd.DataFrame(results)
    return results_df.set_index(["building", "floor"])
