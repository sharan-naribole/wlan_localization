"""Visualization functions for model evaluation and analysis."""

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from numpy.typing import NDArray
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix
from sklearn.model_selection import learning_curve

from wlan_localization.utils.logger import get_logger

logger = get_logger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 8)
plt.rcParams["font.size"] = 10


def plot_roc_curves(
    y_true: NDArray[np.int_],
    y_pred_proba: NDArray[np.float64],
    class_names: List[str],
    title: str = "ROC Curves",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot ROC curves for multiclass classification.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities (n_samples, n_classes)
        class_names: List of class names
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    from sklearn.preprocessing import label_binarize

    n_classes = len(class_names)

    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot ROC curve for each class
    for i in range(n_classes):
        RocCurveDisplay.from_predictions(
            y_true_bin[:, i],
            y_pred_proba[:, i],
            name=class_names[i],
            ax=ax
        )

    # Plot diagonal
    ax.plot([0, 1], [0, 1], "k--", label="Random (AUC=0.5)")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"ROC curves saved to {save_path}")

    return fig


def plot_confusion_matrix(
    y_true: NDArray[np.int_],
    y_pred: NDArray[np.int_],
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[Path] = None,
    normalize: bool = False
) -> plt.Figure:
    """Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        title: Plot title
        save_path: Optional path to save figure
        normalize: Whether to normalize confusion matrix

    Returns:
        Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred, normalize="true" if normalize else None)

    fig, ax = plt.subplots(figsize=(8, 8))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )
    disp.plot(ax=ax, cmap="Blues", values_format=".2f" if normalize else "d")

    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Confusion matrix saved to {save_path}")

    return fig


def plot_learning_curves(
    estimator,
    X: NDArray[np.float64],
    y: NDArray,
    title: str = "Learning Curves",
    cv: int = 5,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot learning curves showing training and validation performance.

    Args:
        estimator: Fitted scikit-learn estimator
        X: Features
        y: Labels
        title: Plot title
        cv: Number of cross-validation folds
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y,
        cv=cv,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring="neg_mean_squared_error"
    )

    # Convert to RMSE
    train_scores = np.sqrt(-train_scores)
    val_scores = np.sqrt(-val_scores)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot training scores
    ax.plot(train_sizes, train_mean, "o-", color="blue", label="Training RMSE")
    ax.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.15,
        color="blue"
    )

    # Plot validation scores
    ax.plot(train_sizes, val_mean, "o-", color="green", label="Validation RMSE")
    ax.fill_between(
        train_sizes,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.15,
        color="green"
    )

    ax.set_xlabel("Number of Training Samples")
    ax.set_ylabel("RMSE")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Learning curves saved to {save_path}")

    return fig


def plot_error_distribution(
    errors: NDArray[np.float64],
    title: str = "Positioning Error Distribution",
    bins: int = 50,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot histogram of positioning errors.

    Args:
        errors: Array of positioning errors (meters)
        title: Plot title
        bins: Number of histogram bins
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(errors, bins=bins, edgecolor="black", alpha=0.7)

    # Add statistics
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    std_error = np.std(errors)

    ax.axvline(mean_error, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_error:.2f}m")
    ax.axvline(median_error, color="green", linestyle="--", linewidth=2, label=f"Median: {median_error:.2f}m")

    ax.set_xlabel("Positioning Error (meters)")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Add text box with statistics
    textstr = f"Mean: {mean_error:.2f}m\nMedian: {median_error:.2f}m\nStd: {std_error:.2f}m"
    ax.text(
        0.95, 0.95, textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Error distribution saved to {save_path}")

    return fig


def plot_spatial_distribution(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    building_id: Optional[int] = None,
    floor_id: Optional[int] = None,
    title: str = "Spatial Position Distribution",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot spatial distribution of true vs predicted positions.

    Args:
        y_true: True positions with LATITUDE, LONGITUDE, BUILDINGID, FLOOR
        y_pred: Predicted positions
        building_id: Optional filter for specific building
        floor_id: Optional filter for specific floor
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    # Filter by building/floor if specified
    mask = pd.Series([True] * len(y_true))
    if building_id is not None:
        mask &= (y_true["BUILDINGID"] == building_id)
        title += f" - Building {building_id}"
    if floor_id is not None:
        mask &= (y_true["FLOOR"] == floor_id)
        title += f", Floor {floor_id}"

    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot true positions
    ax.scatter(
        y_true_filtered["LONGITUDE"],
        y_true_filtered["LATITUDE"],
        c="blue",
        alpha=0.5,
        s=50,
        label="True Position",
        marker="o"
    )

    # Plot predicted positions
    ax.scatter(
        y_pred_filtered["LONGITUDE"],
        y_pred_filtered["LATITUDE"],
        c="red",
        alpha=0.5,
        s=50,
        label="Predicted Position",
        marker="x"
    )

    # Draw lines connecting true and predicted
    for i in range(len(y_true_filtered)):
        ax.plot(
            [y_true_filtered.iloc[i]["LONGITUDE"], y_pred_filtered.iloc[i]["LONGITUDE"]],
            [y_true_filtered.iloc[i]["LATITUDE"], y_pred_filtered.iloc[i]["LATITUDE"]],
            "k-", alpha=0.1, linewidth=0.5
        )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Spatial distribution saved to {save_path}")

    return fig


def plot_per_location_performance(
    metrics_df: pd.DataFrame,
    metric: str = "rmse",
    title: str = "Performance by Building-Floor",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot performance metrics per building-floor combination.

    Args:
        metrics_df: DataFrame with (building, floor) index and metric columns
        metric: Metric column to plot
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create x labels
    x_labels = [f"B{b}F{f}" for b, f in metrics_df.index]
    x_pos = np.arange(len(x_labels))

    # Plot bars
    bars = ax.bar(x_pos, metrics_df[metric], color="steelblue", alpha=0.7, edgecolor="black")

    # Color bars by building
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, (building, _) in enumerate(metrics_df.index):
        bars[i].set_color(colors[building])

    ax.set_xlabel("Building-Floor")
    ax.set_ylabel(f"{metric.upper()} (meters)")
    ax.set_title(title)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for i, (idx, value) in enumerate(metrics_df[metric].items()):
        ax.text(i, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Per-location performance saved to {save_path}")

    return fig


def create_evaluation_report(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    metrics: dict,
    output_dir: Path
) -> None:
    """Create comprehensive evaluation report with all visualizations.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        metrics: Dictionary of evaluation metrics
        output_dir: Directory to save figures
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Creating evaluation report in {output_dir}")

    # 1. Error distribution
    errors = np.sqrt(
        (y_true["LATITUDE"] - y_pred["LATITUDE"]) ** 2 +
        (y_true["LONGITUDE"] - y_pred["LONGITUDE"]) ** 2
    )
    plot_error_distribution(
        errors.values,
        save_path=output_dir / "error_distribution.png"
    )

    # 2. Building confusion matrix
    plot_confusion_matrix(
        y_true["BUILDINGID"].values,
        y_pred["BUILDINGID"].values,
        class_names=["Building 0", "Building 1", "Building 2"],
        title="Building Classification Confusion Matrix",
        save_path=output_dir / "building_confusion_matrix.png",
        normalize=True
    )

    # 3. Floor confusion matrix
    plot_confusion_matrix(
        y_true["FLOOR"].values,
        y_pred["FLOOR"].values,
        class_names=[f"Floor {i}" for i in range(5)],
        title="Floor Classification Confusion Matrix",
        save_path=output_dir / "floor_confusion_matrix.png",
        normalize=True
    )

    # 4. Spatial distributions per building
    for building in y_true["BUILDINGID"].unique():
        plot_spatial_distribution(
            y_true,
            y_pred,
            building_id=building,
            save_path=output_dir / f"spatial_building_{building}.png"
        )

    logger.info("Evaluation report complete")
