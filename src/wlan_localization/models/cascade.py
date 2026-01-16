"""Cascade pipeline orchestrating building→floor→position prediction."""

from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray

from wlan_localization.data.preprocessor import DataPreprocessor
from wlan_localization.evaluation.metrics import evaluate_cascade_pipeline
from wlan_localization.models.building_classifier import BuildingClassifier
from wlan_localization.models.floor_classifier import FloorClassifier
from wlan_localization.models.position_regressor import PositionRegressor
from wlan_localization.utils.logger import get_logger

logger = get_logger(__name__)


class CascadePipeline:
    """Three-stage cascade pipeline for indoor localization.

    Pipeline stages:
        1. Building Classification (Random Forest with class balancing)
        2. Per-Building Floor Classification (Weighted KNN)
        3. Per-Building-Floor Position Regression (Weighted KNN)

    Attributes:
        preprocessor: Data preprocessing pipeline
        building_clf: Building classifier
        floor_clf: Floor classifier
        position_reg: Position regressor
        config: Configuration dictionary
    """

    def __init__(
        self,
        preprocessor: Optional[DataPreprocessor] = None,
        building_clf: Optional[BuildingClassifier] = None,
        floor_clf: Optional[FloorClassifier] = None,
        position_reg: Optional[PositionRegressor] = None,
        config: Optional[dict] = None
    ) -> None:
        """Initialize cascade pipeline.

        Args:
            preprocessor: Data preprocessor (created if None)
            building_clf: Building classifier (created if None)
            floor_clf: Floor classifier (created if None)
            position_reg: Position regressor (created if None)
            config: Configuration dictionary
        """
        self.config = config or self._default_config()

        self.preprocessor = preprocessor or DataPreprocessor(
            **self.config.get("preprocessing", {})
        )

        self.building_clf = building_clf or BuildingClassifier(
            **self.config.get("building_classifier", {})
        )

        self.floor_clf = floor_clf or FloorClassifier(
            **self.config.get("floor_classifier", {})
        )

        self.position_reg = position_reg or PositionRegressor(
            **self.config.get("position_regressor", {})
        )

        self._is_fitted = False

    @staticmethod
    def _default_config() -> dict:
        """Get default configuration.

        Returns:
            Default configuration dictionary
        """
        return {
            "preprocessing": {
                "missing_value": 100.0,
                "apply_box_cox": True,
                "n_components": 150,
                "explained_variance": 0.95
            },
            "building_classifier": {
                "n_estimators": 100,
                "random_state": 42,
                "balancing_strategy": "nearmiss"
            },
            "floor_classifier": {
                "k": 3,
                "metric": "manhattan",
                "weights": "distance"
            },
            "position_regressor": {
                "k": 3,
                "metric": "manhattan",
                "weights": "distance"
            }
        }

    @classmethod
    def from_config(cls, config_path: str) -> "CascadePipeline":
        """Create pipeline from YAML configuration file.

        Args:
            config_path: Path to YAML config file

        Returns:
            Configured CascadePipeline instance

        Example:
            >>> pipeline = CascadePipeline.from_config('configs/cascade_optimal.yaml')
            >>> pipeline.fit(X_train, y_train)
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded configuration from {config_path}")
        return cls(config=config)

    def fit(
        self,
        X: NDArray[np.float64],
        y: pd.DataFrame
    ) -> "CascadePipeline":
        """Fit complete cascade pipeline.

        Args:
            X: Raw RSSI features
            y: DataFrame with columns ['LATITUDE', 'LONGITUDE', 'BUILDINGID', 'FLOOR']

        Returns:
            Self for method chaining
        """
        logger.info("=" * 60)
        logger.info("TRAINING CASCADE PIPELINE")
        logger.info("=" * 60)

        # Stage 0: Preprocessing
        logger.info("\n[Stage 0] Preprocessing")
        X_processed = self.preprocessor.fit_transform(X)

        # Stage 1: Building Classification
        logger.info("\n[Stage 1] Building Classification")
        self.building_clf.fit(X_processed, y["BUILDINGID"].values)

        # Stage 2: Per-Building Floor Classification
        logger.info("\n[Stage 2] Per-Building Floor Classification")
        self.floor_clf.fit(
            X_processed,
            y_floor=y["FLOOR"].values,
            y_building=y["BUILDINGID"].values  # Use true buildings for training
        )

        # Stage 3: Per-Building-Floor Position Regression
        logger.info("\n[Stage 3] Per-Building-Floor Position Regression")
        y_position = y[["LATITUDE", "LONGITUDE"]].values
        self.position_reg.fit(
            X_processed,
            y_position=y_position,
            y_building=y["BUILDINGID"].values,  # Use true for training
            y_floor=y["FLOOR"].values  # Use true for training
        )

        self._is_fitted = True

        logger.info("\n" + "=" * 60)
        logger.info("CASCADE PIPELINE TRAINING COMPLETE")
        logger.info("=" * 60)

        return self

    def predict(
        self,
        X: NDArray[np.float64]
    ) -> pd.DataFrame:
        """Predict building, floor, and position through cascade.

        Args:
            X: Raw RSSI features

        Returns:
            DataFrame with columns ['BUILDINGID', 'FLOOR', 'LATITUDE', 'LONGITUDE']

        Raises:
            RuntimeError: If pipeline not fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")

        # Stage 0: Preprocessing
        X_processed = self.preprocessor.transform(X)

        # Stage 1: Building Classification
        building_pred = self.building_clf.predict(X_processed)

        # Stage 2: Floor Classification (using predicted buildings)
        floor_pred = self.floor_clf.predict(X_processed, building_pred)

        # Stage 3: Position Regression (using predicted building and floor)
        position_pred = self.position_reg.predict(
            X_processed,
            building_pred,
            floor_pred
        )

        # Combine predictions
        predictions = pd.DataFrame({
            "BUILDINGID": building_pred,
            "FLOOR": floor_pred,
            "LATITUDE": position_pred[:, 0],
            "LONGITUDE": position_pred[:, 1]
        })

        return predictions

    def evaluate(
        self,
        X: NDArray[np.float64],
        y: pd.DataFrame,
        building_penalty: float = 50.0,
        floor_penalty: float = 4.0
    ) -> Dict[str, float]:
        """Evaluate cascade pipeline with custom positioning error metric.

        Args:
            X: Raw RSSI features
            y: True labels DataFrame
            building_penalty: Penalty for building misclassification
            floor_penalty: Penalty for floor misclassification

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating cascade pipeline...")

        y_pred = self.predict(X)

        metrics = evaluate_cascade_pipeline(
            y_true=y,
            y_pred=y_pred,
            building_penalty=building_penalty,
            floor_penalty=floor_penalty
        )

        return metrics

    def save(self, path: str) -> None:
        """Save pipeline to disk.

        Args:
            path: Save path (will create directory if needed)
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save each component
        joblib.dump(self.preprocessor, path / "preprocessor.pkl")
        joblib.dump(self.building_clf, path / "building_clf.pkl")
        joblib.dump(self.floor_clf, path / "floor_clf.pkl")
        joblib.dump(self.position_reg, path / "position_reg.pkl")

        # Save config
        with open(path / "config.yaml", "w") as f:
            yaml.dump(self.config, f)

        logger.info(f"Pipeline saved to {path}")

    @classmethod
    def load(cls, path: str) -> "CascadePipeline":
        """Load pipeline from disk.

        Args:
            path: Load path

        Returns:
            Loaded CascadePipeline instance

        Example:
            >>> pipeline = CascadePipeline.load('models/cascade_best/')
            >>> predictions = pipeline.predict(X_test)
        """
        path = Path(path)

        # Load components
        preprocessor = joblib.load(path / "preprocessor.pkl")
        building_clf = joblib.load(path / "building_clf.pkl")
        floor_clf = joblib.load(path / "floor_clf.pkl")
        position_reg = joblib.load(path / "position_reg.pkl")

        # Load config
        with open(path / "config.yaml", "r") as f:
            config = yaml.safe_load(f)

        pipeline = cls(
            preprocessor=preprocessor,
            building_clf=building_clf,
            floor_clf=floor_clf,
            position_reg=position_reg,
            config=config
        )
        pipeline._is_fitted = True

        logger.info(f"Pipeline loaded from {path}")

        return pipeline

    def get_stage_predictions(
        self,
        X: NDArray[np.float64],
        y_true: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Get predictions and metrics for each pipeline stage.

        Useful for debugging and analyzing cascade performance.

        Args:
            X: Raw RSSI features
            y_true: Optional true labels for evaluation

        Returns:
            Dictionary with predictions and metrics per stage
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")

        results = {}

        # Preprocessing
        X_processed = self.preprocessor.transform(X)
        results["X_processed_shape"] = X_processed.shape

        # Stage 1: Building
        building_pred = self.building_clf.predict(X_processed)
        building_proba = self.building_clf.predict_proba(X_processed)
        results["building_pred"] = building_pred
        results["building_proba"] = building_proba

        if y_true is not None:
            building_metrics = self.building_clf.evaluate(
                X_processed, y_true["BUILDINGID"].values
            )
            results["building_metrics"] = building_metrics

        # Stage 2: Floor
        floor_pred = self.floor_clf.predict(X_processed, building_pred)
        results["floor_pred"] = floor_pred

        if y_true is not None:
            floor_metrics = self.floor_clf.evaluate(
                X_processed,
                y_true["FLOOR"].values,
                y_true["BUILDINGID"].values
            )
            results["floor_metrics"] = floor_metrics

        # Stage 3: Position
        position_pred = self.position_reg.predict(
            X_processed, building_pred, floor_pred
        )
        results["position_pred"] = position_pred

        if y_true is not None:
            position_metrics = self.position_reg.evaluate(
                X_processed,
                y_true[["LATITUDE", "LONGITUDE"]].values,
                y_true["BUILDINGID"].values,
                y_true["FLOOR"].values
            )
            results["position_metrics"] = position_metrics

        return results
