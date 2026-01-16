"""Tests for data preprocessing module."""

import numpy as np
import pytest

from wlan_localization.data.preprocessor import DataPreprocessor


class TestDataPreprocessor:
    """Test DataPreprocessor class."""

    def test_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = DataPreprocessor(
            missing_value=100.0,
            apply_box_cox=True,
            n_components=50,
            explained_variance=0.95
        )

        assert preprocessor.missing_value == 100.0
        assert preprocessor.apply_box_cox is True
        assert preprocessor.n_components == 50
        assert preprocessor.explained_variance == 0.95

    def test_handle_missing_values_indicator(self, sample_rssi_data):
        """Test missing value handling with indicator strategy."""
        X, _ = sample_rssi_data
        preprocessor = DataPreprocessor()

        X_processed = preprocessor.handle_missing_values(X, strategy="indicator")

        # Should keep missing values as-is
        assert np.array_equal(X_processed, X)

    def test_handle_missing_values_zero(self, sample_rssi_data):
        """Test missing value handling with zero strategy."""
        X, _ = sample_rssi_data
        preprocessor = DataPreprocessor(missing_value=100.0)

        X_processed = preprocessor.handle_missing_values(X, strategy="zero")

        # Missing values should be replaced with 0
        assert (X_processed[X == 100.0] == 0).all()

    def test_pca_fit_transform(self, sample_rssi_data):
        """Test PCA dimensionality reduction."""
        X, _ = sample_rssi_data
        preprocessor = DataPreprocessor(n_components=50)

        # Fit and transform
        X_transformed = preprocessor.fit_transform(X)

        # Check dimensions
        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] == 50

        # Check PCA is fitted
        assert preprocessor.pca is not None
        assert preprocessor.scaler is not None

    def test_fit_transform_pipeline(self, sample_rssi_data):
        """Test complete preprocessing pipeline."""
        X, _ = sample_rssi_data
        preprocessor = DataPreprocessor(
            apply_box_cox=False,  # Skip Box-Cox for faster testing
            n_components=30
        )

        X_transformed = preprocessor.fit_transform(X)

        # Check output shape
        assert X_transformed.shape == (100, 30)

        # Check data is scaled (mean ~0, std ~1)
        assert np.abs(X_transformed.mean()) < 0.5
        assert np.abs(X_transformed.std() - 1.0) < 0.5

    def test_transform_without_fit_raises_error(self, sample_rssi_data):
        """Test that transform without fit raises error."""
        X, _ = sample_rssi_data
        preprocessor = DataPreprocessor()

        with pytest.raises(RuntimeError, match="not fitted"):
            preprocessor.transform(X)

    def test_get_feature_statistics(self, sample_rssi_data):
        """Test feature statistics computation."""
        X, _ = sample_rssi_data
        preprocessor = DataPreprocessor(missing_value=100.0)

        stats = preprocessor.get_feature_statistics(X)

        # Check statistics are computed
        assert stats["n_samples"] == 100
        assert stats["n_features"] == 520
        assert 0 < stats["sparsity"] < 1
        assert stats["mean_aps_per_sample"] > 0
        assert stats["min_rssi"] < stats["max_rssi"]

    def test_pca_explained_variance(self, sample_rssi_data):
        """Test PCA retains target explained variance."""
        X, _ = sample_rssi_data
        preprocessor = DataPreprocessor(
            apply_box_cox=False,
            n_components=None,
            explained_variance=0.90
        )

        preprocessor.fit(X)

        # Check explained variance
        explained = preprocessor.pca.explained_variance_ratio_.sum()
        assert explained >= 0.90

    def test_fit_pca_auto_components(self, sample_rssi_data):
        """Test automatic component selection based on variance."""
        X, _ = sample_rssi_data
        preprocessor = DataPreprocessor(
            apply_box_cox=False,
            n_components=None,
            explained_variance=0.95
        )

        preprocessor.fit_pca(X)

        # Should select some number of components
        assert preprocessor.n_components is not None
        assert preprocessor.n_components > 0
        assert preprocessor.n_components < X.shape[1]

    def test_transform_consistency(self, sample_rssi_data):
        """Test that transform produces consistent results."""
        X, _ = sample_rssi_data
        preprocessor = DataPreprocessor(n_components=30)

        preprocessor.fit(X)

        # Transform twice
        X_t1 = preprocessor.transform(X)
        X_t2 = preprocessor.transform(X)

        # Should be identical
        np.testing.assert_array_almost_equal(X_t1, X_t2)
