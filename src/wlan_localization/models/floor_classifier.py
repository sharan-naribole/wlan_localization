"""Per-building floor classification using weighted KNN."""

from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from wlan_localization.utils.logger import get_logger

logger = get_logger(__name__)


class FloorClassifier:
    """KNN classifier for floor identification within a building.

    Separate models are trained for each building to handle different
    floor configurations and RF characteristics.

    Attributes:
        k: Number of neighbors for KNN
        metric: Distance metric ('manhattan', 'euclidean', 'minkowski')
        weights: Weight function ('uniform' or 'distance')
        models: Dictionary mapping building_id → fitted pipeline
    """

    def __init__(
        self,
        k: int = 3,
        metric: str = "manhattan",
        weights: str = "distance"
    ) -> None:
        """Initialize floor classifier.

        Args:
            k: Number of neighbors
            metric: Distance metric for KNN
            weights: 'uniform' or 'distance' weighting
        """
        self.k = k
        self.metric = metric
        self.weights = weights

        # Dictionary to store per-building models
        self.models: Dict[int, Pipeline] = {}
        self._building_floors: Dict[int, NDArray[np.int_]] = {}

    def _create_pipeline(self) -> Pipeline:
        """Create KNN pipeline with standard scaling.

        Returns:
            Pipeline with scaler and KNN classifier
        """
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(
                n_neighbors=self.k,
                metric=self.metric,
                weights=self.weights,
                n_jobs=-1
            ))
        ])
        return pipeline

    def fit_building(
        self,
        building_id: int,
        X: NDArray[np.float64],
        y: NDArray[np.int_]
    ) -> None:
        """Fit floor classifier for a specific building.

        Args:
            building_id: Building ID
            X: Training features for this building
            y: Floor IDs for this building
        """
        logger.info(f"Training floor classifier for Building {building_id} on {len(X)} samples")

        # Store floor classes for this building
        self._building_floors[building_id] = np.unique(y)
        logger.info(f"Building {building_id} floors: {self._building_floors[building_id]}")

        # Log class distribution
        unique, counts = np.unique(y, return_counts=True)
        for floor, count in zip(unique, counts):
            logger.info(f"  Floor {floor}: {count} samples ({count/len(y):.1%})")

        # Create and fit pipeline
        pipeline = self._create_pipeline()
        pipeline.fit(X, y)
        self.models[building_id] = pipeline

        # Log training accuracy
        y_pred_train = pipeline.predict(X)
        train_acc = accuracy_score(y, y_pred_train)
        logger.info(f"Building {building_id} training accuracy: {train_acc:.1%}")

    def fit(
        self,
        X: NDArray[np.float64],
        y_floor: NDArray[np.int_],
        y_building: NDArray[np.int_]
    ) -> "FloorClassifier":
        """Fit floor classifiers for all buildings.

        Args:
            X: Training features
            y_floor: Floor IDs
            y_building: Building IDs

        Returns:
            Self for method chaining
        """
        # Train separate model for each building
        for building_id in np.unique(y_building):
            mask = y_building == building_id
            X_building = X[mask]
            y_building_floors = y_floor[mask]

            self.fit_building(building_id, X_building, y_building_floors)

        logger.info(f"Trained {len(self.models)} floor classifiers")
        return self

    def predict(
        self,
        X: NDArray[np.float64],
        building_ids: NDArray[np.int_]
    ) -> NDArray[np.int_]:
        """Predict floor IDs using appropriate building-specific model.

        Args:
            X: Features
            building_ids: Predicted or known building IDs for each sample

        Returns:
            Predicted floor IDs

        Raises:
            RuntimeError: If models not fitted or building ID not found
        """
        if not self.models:
            raise RuntimeError("Models not fitted. Call fit() first.")

        n_samples = len(X)
        y_pred = np.zeros(n_samples, dtype=int)

        # Predict floor for each sample using its building's model
        for building_id in np.unique(building_ids):
            if building_id not in self.models:
                raise RuntimeError(
                    f"No model trained for building {building_id}. "
                    f"Available buildings: {list(self.models.keys())}"
                )

            mask = building_ids == building_id
            if mask.sum() == 0:
                continue

            X_building = X[mask]
            y_pred[mask] = self.models[building_id].predict(X_building)

        return y_pred

    def predict_proba(
        self,
        X: NDArray[np.float64],
        building_ids: NDArray[np.int_]
    ) -> Dict[int, NDArray[np.float64]]:
        """Predict floor probabilities per building.

        Args:
            X: Features
            building_ids: Building IDs

        Returns:
            Dictionary mapping building_id → probability array
        """
        if not self.models:
            raise RuntimeError("Models not fitted. Call fit() first.")

        probas = {}

        for building_id in np.unique(building_ids):
            mask = building_ids == building_id
            if mask.sum() == 0:
                continue

            X_building = X[mask]
            probas[building_id] = self.models[building_id].predict_proba(X_building)

        return probas

    def evaluate(
        self,
        X: NDArray[np.float64],
        y_floor_true: NDArray[np.int_],
        y_building_true: NDArray[np.int_]
    ) -> Dict[str, float]:
        """Evaluate floor classification performance.

        Args:
            X: Features
            y_floor_true: True floor IDs
            y_building_true: True building IDs

        Returns:
            Dictionary with overall and per-building accuracy
        """
        y_pred = self.predict(X, y_building_true)

        # Overall accuracy
        overall_acc = accuracy_score(y_floor_true, y_pred)

        metrics = {
            "overall_accuracy": overall_acc,
            "n_samples": len(X)
        }

        # Per-building accuracy
        for building_id in np.unique(y_building_true):
            mask = y_building_true == building_id
            if mask.sum() > 0:
                building_acc = accuracy_score(
                    y_floor_true[mask],
                    y_pred[mask]
                )
                metrics[f"building_{building_id}_floor_accuracy"] = building_acc

                logger.info(
                    f"Building {building_id} floor accuracy: {building_acc:.1%} "
                    f"({mask.sum()} samples)"
                )

        logger.info(f"Overall floor accuracy: {overall_acc:.1%}")

        return metrics

    def get_building_floors(self, building_id: int) -> Optional[NDArray[np.int_]]:
        """Get list of floors for a specific building.

        Args:
            building_id: Building ID

        Returns:
            Array of floor IDs for this building, or None if building not found
        """
        return self._building_floors.get(building_id, None)
