"""Data preprocessing pipeline for RSSI signals."""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from wlan_localization.utils.logger import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """Preprocessing pipeline for Wi-Fi RSSI data.

    Applies missing value handling, Box-Cox transformation, PCA dimensionality
    reduction, and standard scaling.

    Attributes:
        missing_value: Value indicating out-of-range signal (default: 100)
        box_cox_lambda: Fitted Box-Cox transformation parameter
        pca: Fitted PCA transformer
        scaler: Fitted standard scaler
    """

    def __init__(
        self,
        missing_value: float = 100.0,
        apply_box_cox: bool = True,
        n_components: Optional[int] = 150,
        explained_variance: float = 0.95
    ) -> None:
        """Initialize preprocessor.

        Args:
            missing_value: Value indicating out-of-range signal
            apply_box_cox: Whether to apply Box-Cox transformation
            n_components: Number of PCA components (None for auto based on variance)
            explained_variance: Target explained variance if n_components is None
        """
        self.missing_value = missing_value
        self.apply_box_cox = apply_box_cox
        self.n_components = n_components
        self.explained_variance = explained_variance

        # Fitted transformers
        self.box_cox_lambda: Optional[float] = None
        self.pca: Optional[PCA] = None
        self.scaler: Optional[StandardScaler] = None

        # Statistics
        self._original_dim: Optional[int] = None
        self._reduced_dim: Optional[int] = None

    def handle_missing_values(
        self,
        X: NDArray[np.float64],
        strategy: str = "indicator"
    ) -> NDArray[np.float64]:
        """Handle missing RSSI values (out-of-range signals).

        Args:
            X: RSSI feature array
            strategy: How to handle missing values
                - 'indicator': Keep missing_value as-is (e.g., 100)
                - 'zero': Replace with 0
                - 'mean': Replace with mean of non-missing values

        Returns:
            Array with missing values handled
        """
        if strategy == "indicator":
            # Keep as-is (missing_value indicates out-of-range)
            return X

        elif strategy == "zero":
            X_processed = X.copy()
            X_processed[X == self.missing_value] = 0
            return X_processed

        elif strategy == "mean":
            X_processed = X.copy()
            for col_idx in range(X.shape[1]):
                col = X[:, col_idx]
                mask = col != self.missing_value
                if mask.any():
                    mean_val = col[mask].mean()
                    X_processed[~mask, col_idx] = mean_val
            return X_processed

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def apply_box_cox_transform(
        self,
        X: NDArray[np.float64],
        fit: bool = True
    ) -> NDArray[np.float64]:
        """Apply Box-Cox transformation to address skewness.

        Box-Cox transformation makes the data more normally distributed,
        which can improve model performance.

        Args:
            X: Feature array (must be positive values)
            fit: Whether to fit lambda parameter or use existing

        Returns:
            Transformed array

        Note:
            RSSI values are typically negative. We add offset to make positive.
        """
        if not self.apply_box_cox:
            return X

        # Make values positive (RSSI is negative, missing_value is positive)
        # Shift all values to be positive
        X_positive = X + np.abs(X.min()) + 1

        if fit:
            # Fit Box-Cox on flattened data
            _, self.box_cox_lambda = stats.boxcox(X_positive.flatten() + 1)
            logger.info(f"Fitted Box-Cox lambda: {self.box_cox_lambda:.4f}")

        # Apply transformation
        X_transformed = stats.boxcox(X_positive + 1, lmbda=self.box_cox_lambda)

        return X_transformed.reshape(X.shape)

    def fit_pca(
        self,
        X: NDArray[np.float64]
    ) -> None:
        """Fit PCA dimensionality reduction.

        Args:
            X: Feature array to fit PCA on
        """
        self._original_dim = X.shape[1]

        # Determine number of components
        if self.n_components is None:
            # Auto-select based on explained variance
            pca_temp = PCA(n_components=min(X.shape))
            pca_temp.fit(X)

            cumsum_var = np.cumsum(pca_temp.explained_variance_ratio_)
            n_comp = np.argmax(cumsum_var >= self.explained_variance) + 1
            self.n_components = n_comp

        # Fit PCA
        self.pca = PCA(n_components=self.n_components, random_state=42)
        self.pca.fit(X)

        self._reduced_dim = self.n_components
        explained = self.pca.explained_variance_ratio_.sum()

        logger.info(
            f"PCA: Reduced {self._original_dim} dims → {self._reduced_dim} dims "
            f"(explained variance: {explained:.1%})"
        )

    def transform_pca(
        self,
        X: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Apply fitted PCA transformation.

        Args:
            X: Feature array

        Returns:
            PCA-transformed array

        Raises:
            RuntimeError: If PCA not fitted yet
        """
        if self.pca is None:
            raise RuntimeError("PCA not fitted. Call fit() or fit_pca() first.")

        return self.pca.transform(X)

    def fit_scaler(
        self,
        X: NDArray[np.float64]
    ) -> None:
        """Fit standard scaler for normalization.

        Args:
            X: Feature array
        """
        self.scaler = StandardScaler()
        self.scaler.fit(X)
        logger.info("Fitted standard scaler")

    def transform_scaler(
        self,
        X: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Apply fitted scaler.

        Args:
            X: Feature array

        Returns:
            Scaled array

        Raises:
            RuntimeError: If scaler not fitted yet
        """
        if self.scaler is None:
            raise RuntimeError("Scaler not fitted. Call fit() or fit_scaler() first.")

        return self.scaler.transform(X)

    def fit(
        self,
        X: NDArray[np.float64]
    ) -> "DataPreprocessor":
        """Fit all preprocessing transformations.

        Args:
            X: Training feature array

        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting preprocessor on data shape: {X.shape}")

        # Handle missing values
        X_processed = self.handle_missing_values(X, strategy="indicator")

        # Box-Cox transformation
        if self.apply_box_cox:
            X_processed = self.apply_box_cox_transform(X_processed, fit=True)

        # PCA
        self.fit_pca(X_processed)

        # Transform through PCA for scaler fitting
        X_pca = self.transform_pca(X_processed)

        # Standard scaler
        self.fit_scaler(X_pca)

        logger.info("Preprocessor fitting complete")
        return self

    def transform(
        self,
        X: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Apply all fitted transformations.

        Args:
            X: Feature array

        Returns:
            Fully transformed array ready for ML models

        Raises:
            RuntimeError: If preprocessor not fitted yet
        """
        if self.pca is None or self.scaler is None:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")

        # Handle missing values
        X_processed = self.handle_missing_values(X, strategy="indicator")

        # Box-Cox transformation
        if self.apply_box_cox and self.box_cox_lambda is not None:
            X_processed = self.apply_box_cox_transform(X_processed, fit=False)

        # PCA
        X_pca = self.transform_pca(X_processed)

        # Standard scaling
        X_scaled = self.transform_scaler(X_pca)

        return X_scaled

    def fit_transform(
        self,
        X: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Fit and transform in one step.

        Args:
            X: Training feature array

        Returns:
            Transformed array
        """
        return self.fit(X).transform(X)

    def get_feature_statistics(self, X: NDArray[np.float64]) -> dict:
        """Compute statistics about the RSSI features.

        Args:
            X: Raw RSSI feature array

        Returns:
            Dictionary with statistics
        """
        # Mask for in-range values
        in_range_mask = X != self.missing_value

        stats_dict = {
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "sparsity": 1 - (in_range_mask.sum() / X.size),
            "mean_aps_per_sample": in_range_mask.sum(axis=1).mean(),
            "median_aps_per_sample": np.median(in_range_mask.sum(axis=1)),
            "min_rssi": X[in_range_mask].min() if in_range_mask.any() else None,
            "max_rssi": X[in_range_mask].max() if in_range_mask.any() else None,
            "mean_rssi": X[in_range_mask].mean() if in_range_mask.any() else None,
        }

        return stats_dict
