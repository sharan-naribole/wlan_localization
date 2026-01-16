"""Tests for building classifier."""

import numpy as np
import pytest

from wlan_localization.models.building_classifier import BuildingClassifier


class TestBuildingClassifier:
    """Test BuildingClassifier class."""

    def test_initialization(self):
        """Test classifier initialization."""
        clf = BuildingClassifier(
            n_estimators=50,
            random_state=42,
            balancing_strategy="nearmiss"
        )

        assert clf.n_estimators == 50
        assert clf.random_state == 42
        assert clf.balancing_strategy == "nearmiss"

    def test_fit_predict(self, balanced_classification_data):
        """Test fit and predict."""
        X, y = balanced_classification_data

        clf = BuildingClassifier(
            n_estimators=10,
            random_state=42,
            balancing_strategy=None  # No balancing for faster testing
        )

        clf.fit(X, y)

        # Check model is fitted
        assert clf.model is not None
        assert clf._classes is not None

        # Make predictions
        y_pred = clf.predict(X)

        assert y_pred.shape == y.shape
        assert set(y_pred) <= set([0, 1, 2])

        # Should achieve reasonable accuracy on training data
        accuracy = (y_pred == y).mean()
        assert accuracy > 0.8  # At least 80% accuracy

    def test_predict_proba(self, balanced_classification_data):
        """Test probability predictions."""
        X, y = balanced_classification_data

        clf = BuildingClassifier(n_estimators=10, random_state=42, balancing_strategy=None)
        clf.fit(X, y)

        y_proba = clf.predict_proba(X)

        # Check shape
        assert y_proba.shape == (len(X), 3)  # 3 classes

        # Probabilities should sum to 1
        np.testing.assert_array_almost_equal(y_proba.sum(axis=1), np.ones(len(X)))

        # Probabilities should be in [0, 1]
        assert (y_proba >= 0).all()
        assert (y_proba <= 1).all()

    def test_evaluate(self, balanced_classification_data):
        """Test evaluation method."""
        X, y = balanced_classification_data

        clf = BuildingClassifier(n_estimators=10, random_state=42, balancing_strategy=None)
        clf.fit(X, y)

        metrics = clf.evaluate(X, y)

        # Check metrics are returned
        assert "accuracy" in metrics
        assert "n_samples" in metrics

        # Check per-class metrics
        for building in [0, 1, 2]:
            assert f"building_{building}_accuracy" in metrics

    def test_predict_without_fit_raises_error(self, balanced_classification_data):
        """Test that predict without fit raises error."""
        X, _ = balanced_classification_data

        clf = BuildingClassifier()

        with pytest.raises(RuntimeError, match="not fitted"):
            clf.predict(X)

    def test_fit_with_class_balancing(self, balanced_classification_data):
        """Test training with class balancing."""
        X, y = balanced_classification_data

        # Make data imbalanced
        # Keep all building 0 samples, reduce others
        mask = (
            (y == 0)
            | ((y == 1) & (np.random.rand(len(y)) < 0.3))
            | ((y == 2) & (np.random.rand(len(y)) < 0.3))
        )
        X_imb, y_imb = X[mask], y[mask]

        clf = BuildingClassifier(
            n_estimators=10,
            random_state=42,
            balancing_strategy="nearmiss"
        )

        # Should not raise error
        clf.fit(X_imb, y_imb)

        # Should still make predictions
        y_pred = clf.predict(X_imb)
        assert len(y_pred) == len(y_imb)
