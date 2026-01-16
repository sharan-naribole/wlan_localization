"""Per-building-floor position regression using weighted KNN."""

from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from wlan_localization.utils.logger import get_logger

logger = get_logger(__name__)


class PositionRegressor:
    """KNN regressor for latitude/longitude prediction.

    Separate models are trained for each (building, floor) combination
    to capture location-specific RF characteristics.

    Attributes:
        k: Number of neighbors for KNN
        metric: Distance metric ('manhattan', 'euclidean', 'minkowski')
        weights: Weight function ('uniform' or 'distance')
        models: Dictionary mapping (building_id, floor_id) → fitted pipeline
    """

    def __init__(
        self,
        k: int = 3,
        metric: str = "manhattan",
        weights: str = "distance"
    ) -> None:
        """Initialize position regressor.

        Args:
            k: Number of neighbors
            metric: Distance metric for KNN
            weights: 'uniform' or 'distance' weighting
        """
        self.k = k
        self.metric = metric
        self.weights = weights

        # Dictionary to store per-(building,floor) models
        self.models: Dict[Tuple[int, int], Pipeline] = {}
        self._location_stats: Dict[Tuple[int, int], dict] = {}

    def _create_pipeline(self) -> Pipeline:
        """Create KNN regression pipeline with standard scaling.

        Returns:
            Pipeline with scaler and KNN regressor
        """
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsRegressor(
                n_neighbors=self.k,
                metric=self.metric,
                weights=self.weights,
                n_jobs=-1
            ))
        ])
        return pipeline

    def fit_location(
        self,
        building_id: int,
        floor_id: int,
        X: NDArray[np.float64],
        y: NDArray[np.float64]
    ) -> None:
        """Fit position regressor for a specific (building, floor).

        Args:
            building_id: Building ID
            floor_id: Floor ID
            X: Training features for this location
            y: Position labels (latitude, longitude)
        """
        key = (building_id, floor_id)
        logger.info(
            f"Training position regressor for Building {building_id}, "
            f"Floor {floor_id} on {len(X)} samples"
        )

        # Store statistics about this location
        self._location_stats[key] = {
            "n_samples": len(X),
            "lat_mean": y[:, 0].mean(),
            "lat_std": y[:, 0].std(),
            "lon_mean": y[:, 1].mean(),
            "lon_std": y[:, 1].std(),
        }

        # Create and fit pipeline
        pipeline = self._create_pipeline()
        pipeline.fit(X, y)
        self.models[key] = pipeline

        # Log training RMSE
        y_pred_train = pipeline.predict(X)
        rmse = np.sqrt(mean_squared_error(y, y_pred_train))
        logger.info(f"Building {building_id}, Floor {floor_id} training RMSE: {rmse:.2f}m")

    def fit(
        self,
        X: NDArray[np.float64],
        y_position: NDArray[np.float64],
        y_building: NDArray[np.int_],
        y_floor: NDArray[np.int_]
    ) -> "PositionRegressor":
        """Fit position regressors for all (building, floor) combinations.

        Args:
            X: Training features
            y_position: Position labels, shape (n_samples, 2) for [lat, lon]
            y_building: Building IDs
            y_floor: Floor IDs

        Returns:
            Self for method chaining
        """
        # Train separate model for each (building, floor) combination
        for building_id in np.unique(y_building):
            building_mask = y_building == building_id
            floors_in_building = np.unique(y_floor[building_mask])

            for floor_id in floors_in_building:
                mask = building_mask & (y_floor == floor_id)

                if mask.sum() < self.k:
                    logger.warning(
                        f"Skipping Building {building_id}, Floor {floor_id}: "
                        f"only {mask.sum()} samples (< k={self.k})"
                    )
                    continue

                X_location = X[mask]
                y_location = y_position[mask]

                self.fit_location(building_id, floor_id, X_location, y_location)

        logger.info(f"Trained {len(self.models)} position regressors")
        return self

    def predict(
        self,
        X: NDArray[np.float64],
        building_ids: NDArray[np.int_],
        floor_ids: NDArray[np.int_]
    ) -> NDArray[np.float64]:
        """Predict positions using appropriate (building, floor)-specific model.

        Args:
            X: Features
            building_ids: Building IDs for each sample
            floor_ids: Floor IDs for each sample

        Returns:
            Predicted positions, shape (n_samples, 2) for [lat, lon]

        Raises:
            RuntimeError: If models not fitted or location key not found
        """
        if not self.models:
            raise RuntimeError("Models not fitted. Call fit() first.")

        n_samples = len(X)
        y_pred = np.zeros((n_samples, 2))

        # Predict position for each sample using its (building, floor) model
        for building_id in np.unique(building_ids):
            building_mask = building_ids == building_id
            floors_in_building = np.unique(floor_ids[building_mask])

            for floor_id in floors_in_building:
                key = (building_id, floor_id)

                if key not in self.models:
                    # Fallback: use mean position from training if available
                    if key in self._location_stats:
                        logger.warning(
                            f"No model for Building {building_id}, Floor {floor_id}. "
                            f"Using mean position."
                        )
                        mask = building_mask & (floor_ids == floor_id)
                        y_pred[mask, 0] = self._location_stats[key]["lat_mean"]
                        y_pred[mask, 1] = self._location_stats[key]["lon_mean"]
                    else:
                        raise RuntimeError(
                            f"No model or statistics for Building {building_id}, "
                            f"Floor {floor_id}. Available: {list(self.models.keys())}"
                        )
                else:
                    mask = building_mask & (floor_ids == floor_id)
                    if mask.sum() == 0:
                        continue

                    X_location = X[mask]
                    y_pred[mask] = self.models[key].predict(X_location)

        return y_pred

    def evaluate(
        self,
        X: NDArray[np.float64],
        y_position_true: NDArray[np.float64],
        y_building_true: NDArray[np.int_],
        y_floor_true: NDArray[np.int_]
    ) -> Dict[str, float]:
        """Evaluate position regression performance.

        Args:
            X: Features
            y_position_true: True positions [lat, lon]
            y_building_true: True building IDs
            y_floor_true: True floor IDs

        Returns:
            Dictionary with overall and per-location RMSE
        """
        y_pred = self.predict(X, y_building_true, y_floor_true)

        # Overall RMSE
        overall_rmse = np.sqrt(mean_squared_error(y_position_true, y_pred))

        metrics = {
            "overall_rmse": overall_rmse,
            "n_samples": len(X)
        }

        # Per-location RMSE
        for building_id in np.unique(y_building_true):
            building_mask = y_building_true == building_id
            floors_in_building = np.unique(y_floor_true[building_mask])

            for floor_id in floors_in_building:
                mask = building_mask & (y_floor_true == floor_id)

                if mask.sum() == 0:
                    continue

                rmse = np.sqrt(mean_squared_error(
                    y_position_true[mask],
                    y_pred[mask]
                ))

                key = f"building_{building_id}_floor_{floor_id}_rmse"
                metrics[key] = rmse

                logger.info(
                    f"Building {building_id}, Floor {floor_id} RMSE: {rmse:.2f}m "
                    f"({mask.sum()} samples)"
                )

        logger.info(f"Overall position RMSE: {overall_rmse:.2f}m")

        return metrics

    def get_location_stats(
        self,
        building_id: int,
        floor_id: int
    ) -> Optional[dict]:
        """Get statistics for a specific location.

        Args:
            building_id: Building ID
            floor_id: Floor ID

        Returns:
            Dictionary with location statistics, or None if not available
        """
        key = (building_id, floor_id)
        return self._location_stats.get(key, None)

    def get_trained_locations(self) -> list:
        """Get list of (building, floor) combinations with trained models.

        Returns:
            List of (building_id, floor_id) tuples
        """
        return list(self.models.keys())
