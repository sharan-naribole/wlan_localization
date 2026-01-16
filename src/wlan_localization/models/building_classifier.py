"""Building classification model with class balancing."""

from typing import Dict, Optional

import numpy as np
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import NearMiss
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from wlan_localization.utils.logger import get_logger

logger = get_logger(__name__)


class BuildingClassifier:
    """Random Forest classifier for building identification with class balancing.

    Uses NearMiss undersampling to handle class imbalance, followed by
    Random Forest classification.

    Attributes:
        n_estimators: Number of trees in Random Forest
        random_state: Random seed for reproducibility
        balancing_strategy: Class balancing method ('nearmiss', 'smote', or None)
        model: Fitted imbalanced-learn pipeline
    """

    def __init__(
        self,
        n_estimators: int = 100,
        random_state: int = 42,
        balancing_strategy: str = "nearmiss",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1
    ) -> None:
        """Initialize building classifier.

        Args:
            n_estimators: Number of trees in Random Forest
            random_state: Random seed
            balancing_strategy: 'nearmiss', 'smote', or None
            max_depth: Maximum depth of trees (None for unlimited)
            min_samples_split: Minimum samples required to split
            min_samples_leaf: Minimum samples required at leaf
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.balancing_strategy = balancing_strategy
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        self.model: Optional[object] = None
        self._classes: Optional[NDArray[np.int_]] = None

    def _create_pipeline(self) -> object:
        """Create imbalanced-learn pipeline with balancing and classifier.

        Returns:
            Pipeline object
        """
        # Create Random Forest classifier
        rf_clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=-1
        )

        # Create pipeline with optional class balancing
        if self.balancing_strategy == "nearmiss":
            # NearMiss-2 undersampling
            balancer = NearMiss(version=2, random_state=self.random_state)
            pipeline = make_pipeline(balancer, rf_clf)
            logger.info("Created pipeline with NearMiss undersampling")

        elif self.balancing_strategy == "smote":
            # SMOTE oversampling
            from imblearn.over_sampling import SMOTE
            balancer = SMOTE(random_state=self.random_state)
            pipeline = make_pipeline(balancer, rf_clf)
            logger.info("Created pipeline with SMOTE oversampling")

        else:
            # No balancing
            pipeline = rf_clf
            logger.info("Created Random Forest without class balancing")

        return pipeline

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.int_]
    ) -> "BuildingClassifier":
        """Fit building classifier.

        Args:
            X: Training features
            y: Building IDs (0, 1, 2)

        Returns:
            Self for method chaining
        """
        logger.info(f"Training building classifier on {len(X)} samples")

        # Store unique classes
        self._classes = np.unique(y)
        logger.info(f"Classes: {self._classes}")

        # Log class distribution
        unique, counts = np.unique(y, return_counts=True)
        for building, count in zip(unique, counts):
            logger.info(f"Building {building}: {count} samples ({count/len(y):.1%})")

        # Create and fit pipeline
        self.model = self._create_pipeline()
        self.model.fit(X, y)

        # Log training accuracy
        y_pred_train = self.model.predict(X)
        train_acc = accuracy_score(y, y_pred_train)
        logger.info(f"Training accuracy: {train_acc:.1%}")

        return self

    def predict(
        self,
        X: NDArray[np.float64]
    ) -> NDArray[np.int_]:
        """Predict building IDs.

        Args:
            X: Features

        Returns:
            Predicted building IDs

        Raises:
            RuntimeError: If model not fitted
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        return self.model.predict(X)

    def predict_proba(
        self,
        X: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Predict class probabilities.

        Args:
            X: Features

        Returns:
            Class probabilities (n_samples, n_classes)

        Raises:
            RuntimeError: If model not fitted
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        return self.model.predict_proba(X)

    def evaluate(
        self,
        X: NDArray[np.float64],
        y_true: NDArray[np.int_],
        return_report: bool = False
    ) -> Dict[str, float]:
        """Evaluate model performance.

        Args:
            X: Features
            y_true: True building IDs
            return_report: Whether to print detailed classification report

        Returns:
            Dictionary with accuracy and per-class metrics
        """
        y_pred = self.predict(X)

        accuracy = accuracy_score(y_true, y_pred)

        metrics = {
            "accuracy": accuracy,
            "n_samples": len(X)
        }

        # Per-class accuracy
        for building in self._classes:
            mask = y_true == building
            if mask.sum() > 0:
                class_acc = accuracy_score(y_true[mask], y_pred[mask])
                metrics[f"building_{building}_accuracy"] = class_acc

        logger.info(f"Building classifier accuracy: {accuracy:.1%}")

        if return_report:
            report = classification_report(
                y_true, y_pred,
                target_names=[f"Building {i}" for i in self._classes]
            )
            logger.info(f"\nClassification Report:\n{report}")

        return metrics

    def get_feature_importances(self) -> Optional[NDArray[np.float64]]:
        """Get feature importances from Random Forest.

        Returns:
            Feature importances array, or None if model not fitted

        Note:
            Only available if balancing_strategy is None (direct RF)
        """
        if self.model is None:
            return None

        if isinstance(self.model, RandomForestClassifier):
            return self.model.feature_importances_
        else:
            # Pipeline - extract RF from pipeline
            try:
                rf = self.model.named_steps['randomforestclassifier']
                return rf.feature_importances_
            except (AttributeError, KeyError):
                logger.warning("Cannot extract feature importances from pipeline")
                return None
