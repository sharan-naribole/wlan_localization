"""Integration tests for complete cascade pipeline."""

import pytest

from wlan_localization import CascadePipeline
from wlan_localization.data.preprocessor import DataPreprocessor
from wlan_localization.models.building_classifier import BuildingClassifier
from wlan_localization.models.floor_classifier import FloorClassifier
from wlan_localization.models.position_regressor import PositionRegressor


class TestCascadePipelineIntegration:
    """Integration tests for CascadePipeline."""

    def test_pipeline_initialization(self):
        """Test pipeline initialization with default config."""
        pipeline = CascadePipeline()

        assert pipeline.preprocessor is not None
        assert pipeline.building_clf is not None
        assert pipeline.floor_clf is not None
        assert pipeline.position_reg is not None
        assert not pipeline._is_fitted

    def test_pipeline_initialization_custom_components(self):
        """Test pipeline initialization with custom components."""
        preprocessor = DataPreprocessor(n_components=30)
        building_clf = BuildingClassifier(n_estimators=10)
        floor_clf = FloorClassifier(k=5)
        position_reg = PositionRegressor(k=5)

        pipeline = CascadePipeline(
            preprocessor=preprocessor,
            building_clf=building_clf,
            floor_clf=floor_clf,
            position_reg=position_reg
        )

        assert pipeline.preprocessor.n_components == 30
        assert pipeline.building_clf.n_estimators == 10
        assert pipeline.floor_clf.k == 5
        assert pipeline.position_reg.k == 5

    def test_fit_predict_pipeline(self, small_rssi_data):
        """Test complete fit-predict pipeline."""
        X, y = small_rssi_data

        pipeline = CascadePipeline(config={
            "preprocessing": {
                "apply_box_cox": False,
                "n_components": 10
            },
            "building_classifier": {
                "n_estimators": 5,
                "balancing_strategy": None
            },
            "floor_classifier": {
                "k": 2
            },
            "position_regressor": {
                "k": 2
            }
        })

        # Fit
        pipeline.fit(X, y)
        assert pipeline._is_fitted

        # Predict
        predictions = pipeline.predict(X)

        # Check predictions structure
        assert len(predictions) == len(X)
        assert "BUILDINGID" in predictions.columns
        assert "FLOOR" in predictions.columns
        assert "LATITUDE" in predictions.columns
        assert "LONGITUDE" in predictions.columns

        # Check predictions are valid
        assert predictions["BUILDINGID"].isin([0, 1, 2]).all()
        assert predictions["FLOOR"].isin([0, 1, 2, 3, 4]).all()

    def test_evaluate_pipeline(self, small_rssi_data):
        """Test pipeline evaluation."""
        X, y = small_rssi_data

        pipeline = CascadePipeline(config={
            "preprocessing": {
                "apply_box_cox": False,
                "n_components": 10
            },
            "building_classifier": {
                "n_estimators": 5,
                "balancing_strategy": None
            },
            "floor_classifier": {"k": 2},
            "position_regressor": {"k": 2}
        })

        pipeline.fit(X, y)

        # Evaluate
        metrics = pipeline.evaluate(X, y)

        # Check metrics are returned
        assert "building_accuracy" in metrics
        assert "floor_accuracy" in metrics
        assert "mean_positioning_error" in metrics

    def test_save_load_pipeline(self, small_rssi_data, temp_model_dir):
        """Test saving and loading pipeline."""
        X, y = small_rssi_data

        # Create and train pipeline
        pipeline = CascadePipeline(config={
            "preprocessing": {"apply_box_cox": False, "n_components": 10},
            "building_classifier": {"n_estimators": 5, "balancing_strategy": None},
            "floor_classifier": {"k": 2},
            "position_regressor": {"k": 2}
        })

        pipeline.fit(X, y)

        # Get predictions before saving
        pred_before = pipeline.predict(X)

        # Save
        save_path = temp_model_dir / "test_pipeline"
        pipeline.save(str(save_path))

        # Load
        loaded_pipeline = CascadePipeline.load(str(save_path))

        # Check loaded pipeline is fitted
        assert loaded_pipeline._is_fitted

        # Get predictions after loading
        pred_after = loaded_pipeline.predict(X)

        # Predictions should be identical
        import numpy as np
        np.testing.assert_array_almost_equal(
            pred_before.values,
            pred_after.values
        )

    def test_predict_without_fit_raises_error(self, small_rssi_data):
        """Test that predict without fit raises error."""
        X, _ = small_rssi_data

        pipeline = CascadePipeline()

        with pytest.raises(RuntimeError, match="not fitted"):
            pipeline.predict(X)

    def test_get_stage_predictions(self, small_rssi_data):
        """Test getting predictions for each stage."""
        X, y = small_rssi_data

        pipeline = CascadePipeline(config={
            "preprocessing": {"apply_box_cox": False, "n_components": 10},
            "building_classifier": {"n_estimators": 5, "balancing_strategy": None},
            "floor_classifier": {"k": 2},
            "position_regressor": {"k": 2}
        })

        pipeline.fit(X, y)

        # Get stage predictions
        results = pipeline.get_stage_predictions(X, y_true=y)

        # Check results structure
        assert "X_processed_shape" in results
        assert "building_pred" in results
        assert "floor_pred" in results
        assert "position_pred" in results
        assert "building_metrics" in results
        assert "floor_metrics" in results
        assert "position_metrics" in results

    def test_pipeline_with_minimal_data(self):
        """Test pipeline with minimal viable dataset."""
        import numpy as np
        import pandas as pd

        # Very small dataset
        np.random.seed(42)
        n_samples = 15
        n_features = 20

        X = np.random.randn(n_samples, n_features)
        y = pd.DataFrame({
            "LATITUDE": np.random.uniform(100, 200, n_samples),
            "LONGITUDE": np.random.uniform(300, 400, n_samples),
            "BUILDINGID": np.array([0] * 5 + [1] * 5 + [2] * 5),
            "FLOOR": np.array([0, 1] * 7 + [2]),
        })

        pipeline = CascadePipeline(config={
            "preprocessing": {"apply_box_cox": False, "n_components": 5},
            "building_classifier": {"n_estimators": 3, "balancing_strategy": None},
            "floor_classifier": {"k": 1},
            "position_regressor": {"k": 1}
        })

        # Should complete without error
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)

        assert len(predictions) == len(X)
