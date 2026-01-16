"""Data loading utilities for UJIIndoorLoc dataset."""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from wlan_localization.utils.logger import get_logger

logger = get_logger(__name__)


class UJIIndoorLocDataset:
    """UJIIndoorLoc dataset loader and manager.

    The UJIIndoorLoc dataset contains Wi-Fi fingerprints for indoor localization
    across 3 buildings with multiple floors.

    Attributes:
        n_aps: Number of Wi-Fi Access Points (520)
        missing_value: Value indicating out-of-range signal (default: 100)
        buildings: List of building IDs (0, 1, 2)
        max_floors: Maximum number of floors (5)
    """

    N_APS = 520
    MISSING_VALUE = 100
    BUILDINGS = [0, 1, 2]
    MAX_FLOORS = 5

    # Column name constants
    WAP_PREFIX = "WAP"
    LONGITUDE = "LONGITUDE"
    LATITUDE = "LATITUDE"
    FLOOR = "FLOOR"
    BUILDINGID = "BUILDINGID"
    SPACEID = "SPACEID"
    RELATIVEPOSITION = "RELATIVEPOSITION"
    USERID = "USERID"
    PHONEID = "PHONEID"
    TIMESTAMP = "TIMESTAMP"

    def __init__(self, data_dir: str = "data/raw") -> None:
        """Initialize dataset loader.

        Args:
            data_dir: Directory containing the raw CSV files
        """
        self.data_dir = Path(data_dir)
        self.training_file = self.data_dir / "trainingData.csv"
        self.validation_file = self.data_dir / "validationData.csv"

    def load_raw_data(
        self,
        file_type: str = "training"
    ) -> pd.DataFrame:
        """Load raw UJIIndoorLoc data from CSV.

        Args:
            file_type: Either 'training' or 'validation'

        Returns:
            DataFrame with RSSI values and location labels

        Raises:
            FileNotFoundError: If data files don't exist
            ValueError: If invalid file_type specified
        """
        if file_type == "training":
            filepath = self.training_file
        elif file_type == "validation":
            filepath = self.validation_file
        else:
            raise ValueError(f"Invalid file_type: {file_type}. Use 'training' or 'validation'")

        if not filepath.exists():
            raise FileNotFoundError(
                f"Data file not found: {filepath}. "
                f"Please run 'python scripts/download_data.py' first."
            )

        logger.info(f"Loading {file_type} data from {filepath}")
        data = pd.read_csv(filepath)
        logger.info(f"Loaded {len(data)} samples with {len(data.columns)} columns")

        return data

    def split_features_labels(
        self,
        data: pd.DataFrame,
        include_metadata: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into features (RSSI) and labels (location).

        Args:
            data: Raw UJIIndoorLoc dataframe
            include_metadata: Whether to include USERID, PHONEID, TIMESTAMP in labels

        Returns:
            Tuple of (features_df, labels_df)
                features_df: RSSI values from all WAPs
                labels_df: LONGITUDE, LATITUDE, FLOOR, BUILDINGID, and optionally metadata
        """
        # WAP columns (RSSI features)
        wap_columns = [col for col in data.columns if col.startswith(self.WAP_PREFIX)]
        X = data[wap_columns]

        # Location labels
        label_columns = [self.LONGITUDE, self.LATITUDE, self.FLOOR, self.BUILDINGID]

        if include_metadata:
            label_columns.extend([
                self.SPACEID,
                self.RELATIVEPOSITION,
                self.USERID,
                self.PHONEID,
                self.TIMESTAMP
            ])

        y = data[label_columns]

        logger.info(f"Split into {X.shape[1]} features and {y.shape[1]} labels")

        return X, y

    def remove_invalid_samples(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        min_aps_detected: int = 1
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Remove samples with too few detected APs.

        Args:
            X: Feature dataframe (RSSI values)
            y: Label dataframe
            min_aps_detected: Minimum number of APs that must be in range

        Returns:
            Tuple of (X_clean, y_clean) with invalid samples removed
        """
        # Count how many APs are in range for each sample (value != missing_value)
        n_aps_detected = (X != self.MISSING_VALUE).sum(axis=1)

        # Keep only samples with at least min_aps_detected APs
        valid_mask = n_aps_detected >= min_aps_detected

        n_removed = (~valid_mask).sum()
        if n_removed > 0:
            logger.warning(
                f"Removing {n_removed} samples with fewer than "
                f"{min_aps_detected} APs detected"
            )

        X_clean = X[valid_mask].reset_index(drop=True)
        y_clean = y[valid_mask].reset_index(drop=True)

        return X_clean, y_clean

    def remove_zero_variance_features(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        """Remove WAP columns with zero variance (all values the same).

        Args:
            X: Feature dataframe (RSSI values)

        Returns:
            DataFrame with zero-variance columns removed
        """
        # Identify columns where all values are the same
        variance = X.var()
        zero_var_cols = variance[variance == 0].index.tolist()

        if len(zero_var_cols) > 0:
            logger.info(f"Removing {len(zero_var_cols)} zero-variance WAP columns")
            X = X.drop(columns=zero_var_cols)

        return X

    def create_train_holdout_split(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        test_size: float = 0.1,
        random_state: int = 42,
        stratify_by: Optional[str] = "BUILDINGID"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create train/holdout split with optional stratification.

        Args:
            X: Feature dataframe
            y: Label dataframe
            test_size: Proportion of data to use for holdout set
            random_state: Random seed for reproducibility
            stratify_by: Column name to stratify split (e.g., 'BUILDINGID')

        Returns:
            Tuple of (X_train, X_holdout, y_train, y_holdout)
        """
        stratify = y[stratify_by] if stratify_by and stratify_by in y.columns else None

        X_train, X_holdout, y_train, y_holdout = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )

        logger.info(
            f"Created train/holdout split: "
            f"{len(X_train)} train, {len(X_holdout)} holdout"
        )

        return X_train, X_holdout, y_train, y_holdout


def load_ujiindoorloc(
    data_dir: str = "data/raw",
    test_size: float = 0.1,
    random_state: int = 42,
    remove_invalid: bool = True,
    min_aps_detected: int = 1,
    remove_zero_var: bool = True
) -> Tuple[NDArray[np.float64], pd.DataFrame, NDArray[np.float64], pd.DataFrame]:
    """Convenience function to load and prepare UJIIndoorLoc dataset.

    Args:
        data_dir: Directory containing raw CSV files
        test_size: Proportion for holdout set
        random_state: Random seed
        remove_invalid: Whether to remove samples with too few APs
        min_aps_detected: Minimum APs required if remove_invalid=True
        remove_zero_var: Whether to remove zero-variance features

    Returns:
        Tuple of (X_train, y_train, X_holdout, y_holdout)
            X arrays are numpy arrays, y are pandas DataFrames

    Example:
        >>> X_train, y_train, X_test, y_test = load_ujiindoorloc()
        >>> print(f"Training samples: {len(X_train)}")
        >>> print(f"Features: {X_train.shape[1]}")
    """
    dataset = UJIIndoorLocDataset(data_dir=data_dir)

    # Load training data
    data = dataset.load_raw_data(file_type="training")

    # Split features and labels
    X, y = dataset.split_features_labels(data, include_metadata=False)

    # Remove invalid samples
    if remove_invalid:
        X, y = dataset.remove_invalid_samples(X, y, min_aps_detected=min_aps_detected)

    # Remove zero-variance features
    if remove_zero_var:
        X = dataset.remove_zero_variance_features(X)

    # Create train/holdout split
    X_train, X_holdout, y_train, y_holdout = dataset.create_train_holdout_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify_by="BUILDINGID"
    )

    logger.info("Dataset loading complete")
    logger.info(f"Training set: {X_train.shape}")
    logger.info(f"Holdout set: {X_holdout.shape}")

    return X_train.values, y_train, X_holdout.values, y_holdout
