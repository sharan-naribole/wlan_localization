"""Tests for evaluation metrics."""

import numpy as np
import pandas as pd
import pytest

from wlan_localization.evaluation.metrics import (
    compute_positioning_error,
    compute_rmse,
    evaluate_cascade_pipeline,
    evaluate_per_building_floor,
)


class TestPositioningError:
    """Test positioning error computation."""

    def test_perfect_prediction(self):
        """Test error when prediction is perfect."""
        error = compute_positioning_error(
            actual_lat=100.0,
            actual_lon=200.0,
            actual_building=0,
            actual_floor=1,
            pred_lat=100.0,
            pred_lon=200.0,
            pred_building=0,
            pred_floor=1
        )

        assert error == 0.0

    def test_position_error_only(self):
        """Test error with only position error, no classification error."""
        error = compute_positioning_error(
            actual_lat=100.0,
            actual_lon=200.0,
            actual_building=0,
            actual_floor=1,
            pred_lat=103.0,  # 3m off in lat
            pred_lon=204.0,  # 4m off in lon
            pred_building=0,  # Correct
            pred_floor=1      # Correct
        )

        # Euclidean distance: sqrt(3^2 + 4^2) = 5.0
        assert error == pytest.approx(5.0)

    def test_building_error_penalty(self):
        """Test building misclassification penalty."""
        error = compute_positioning_error(
            actual_lat=100.0,
            actual_lon=200.0,
            actual_building=0,
            actual_floor=1,
            pred_lat=100.0,
            pred_lon=200.0,
            pred_building=1,  # Wrong building
            pred_floor=1,
            building_penalty=50.0,
            floor_penalty=4.0
        )

        # Only building penalty
        assert error == 50.0

    def test_floor_error_penalty(self):
        """Test floor misclassification penalty."""
        error = compute_positioning_error(
            actual_lat=100.0,
            actual_lon=200.0,
            actual_building=0,
            actual_floor=1,
            pred_lat=100.0,
            pred_lon=200.0,
            pred_building=0,
            pred_floor=2,  # Wrong floor
            building_penalty=50.0,
            floor_penalty=4.0
        )

        # Only floor penalty
        assert error == 4.0

    def test_combined_errors(self):
        """Test combined position and classification errors."""
        error = compute_positioning_error(
            actual_lat=100.0,
            actual_lon=200.0,
            actual_building=0,
            actual_floor=1,
            pred_lat=103.0,  # 5m euclidean
            pred_lon=204.0,
            pred_building=1,  # Wrong building (50m penalty)
            pred_floor=2,     # Wrong floor (4m penalty)
            building_penalty=50.0,
            floor_penalty=4.0
        )

        # 5m + 50m + 4m = 59m
        assert error == pytest.approx(59.0)


class TestRMSE:
    """Test RMSE computation."""

    def test_rmse_perfect(self):
        """Test RMSE with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])

        rmse = compute_rmse(y_true, y_pred)
        assert rmse == 0.0

    def test_rmse_with_error(self):
        """Test RMSE with some error."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.5, 2.5, 3.5, 4.5])  # 0.5 error each

        rmse = compute_rmse(y_true, y_pred)
        assert rmse == pytest.approx(0.5)

    def test_rmse_multivariate(self):
        """Test RMSE with multivariate output."""
        y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_pred = np.array([[1.5, 2.5], [3.5, 4.5]])

        rmse = compute_rmse(y_true, y_pred)
        assert rmse > 0


class TestCascadeEvaluation:
    """Test cascade pipeline evaluation."""

    def test_evaluate_cascade_pipeline(self, sample_predictions):
        """Test complete cascade evaluation."""
        y_true, y_pred = sample_predictions

        metrics = evaluate_cascade_pipeline(
            y_true=y_true,
            y_pred=y_pred,
            building_penalty=50.0,
            floor_penalty=4.0
        )

        # Check required metrics are present
        assert "building_accuracy" in metrics
        assert "floor_accuracy" in metrics
        assert "mean_positioning_error" in metrics
        assert "mean_euclidean_distance" in metrics
        assert "std_positioning_error" in metrics

        # Check values are reasonable
        assert 0 <= metrics["building_accuracy"] <= 1
        assert 0 <= metrics["floor_accuracy"] <= 1
        assert metrics["mean_positioning_error"] >= 0
        assert metrics["mean_euclidean_distance"] >= 0

        # Check per-building metrics exist
        for building in [0, 1, 2]:
            key = f"building_{building}_floor_accuracy"
            if key in metrics:
                assert 0 <= metrics[key] <= 1

    def test_per_building_floor_evaluation(self, sample_predictions):
        """Test per-(building, floor) evaluation."""
        y_true, y_pred = sample_predictions

        results_df = evaluate_per_building_floor(y_true, y_pred)

        # Check dataframe structure
        assert isinstance(results_df, pd.DataFrame)
        assert "rmse" in results_df.columns
        assert "mae" in results_df.columns
        assert "n_samples" in results_df.columns

        # Check all values are non-negative
        assert (results_df["rmse"] >= 0).all()
        assert (results_df["mae"] >= 0).all()
        assert (results_df["n_samples"] > 0).all()
