"""CLI for training cascade pipeline."""

import json
from pathlib import Path
from typing import Optional

import click
import mlflow
import yaml

from wlan_localization import CascadePipeline
from wlan_localization.data import load_ujiindoorloc
from wlan_localization.utils.logger import get_logger

logger = get_logger(__name__)


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    default="configs/cascade_optimal.yaml",
    help="Path to configuration YAML file",
)
@click.option(
    "--data-dir",
    type=click.Path(exists=True, path_type=Path),
    default="data/raw",
    help="Directory containing training data",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default="models/cascade_trained",
    help="Output directory for trained model",
)
@click.option(
    "--test-size",
    type=float,
    default=0.1,
    help="Proportion of data to use for holdout validation",
)
@click.option(
    "--random-state",
    type=int,
    default=42,
    help="Random seed for reproducibility",
)
@click.option(
    "--mlflow/--no-mlflow",
    default=True,
    help="Enable MLflow experiment tracking",
)
@click.option(
    "--experiment-name",
    type=str,
    default="wlan_localization",
    help="MLflow experiment name",
)
@click.option(
    "--run-name",
    type=str,
    default=None,
    help="MLflow run name (default: auto-generated)",
)
def main(
    config: Path,
    data_dir: Path,
    output: Path,
    test_size: float,
    random_state: int,
    mlflow: bool,
    experiment_name: str,
    run_name: Optional[str],
) -> None:
    """Train cascade pipeline for indoor localization.

    Example:
        wlan-train --config configs/cascade_optimal.yaml --output models/my_model
    """
    logger.info("=" * 80)
    logger.info("WLAN INDOOR LOCALIZATION - TRAINING")
    logger.info("=" * 80)

    # Load configuration
    logger.info(f"Loading configuration from {config}")
    with open(config, "r") as f:
        config_dict = yaml.safe_load(f)

    # Initialize MLflow if enabled
    if mlflow:
        mlflow.set_experiment(experiment_name)
        run = mlflow.start_run(run_name=run_name)
        logger.info(f"MLflow tracking enabled - Run ID: {run.info.run_id}")

        # Log parameters
        mlflow.log_params({
            "config_file": str(config),
            "test_size": test_size,
            "random_state": random_state,
            **_flatten_dict(config_dict.get("models", {})),
            **_flatten_dict(config_dict.get("preprocessing", {})),
        })

    try:
        # Load data
        logger.info(f"Loading data from {data_dir}")
        X_train, y_train, X_holdout, y_holdout = load_ujiindoorloc(
            data_dir=str(data_dir),
            test_size=test_size,
            random_state=random_state,
            remove_invalid=True,
            min_aps_detected=1,
            remove_zero_var=True,
        )

        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Holdout samples: {len(X_holdout)}")
        logger.info(f"Features: {X_train.shape[1]}")

        # Create and train pipeline
        logger.info("Initializing cascade pipeline")
        pipeline = CascadePipeline.from_config(str(config))

        logger.info("Training pipeline...")
        pipeline.fit(X_train, y_train)

        # Evaluate on training data
        logger.info("\nEvaluating on training data:")
        train_metrics = pipeline.evaluate(X_train, y_train)
        _log_metrics(train_metrics, prefix="train_")

        # Evaluate on holdout data
        logger.info("\nEvaluating on holdout data:")
        holdout_metrics = pipeline.evaluate(X_holdout, y_holdout)
        _log_metrics(holdout_metrics, prefix="holdout_")

        # Log metrics to MLflow
        if mlflow:
            mlflow.log_metrics({
                **{f"train_{k}": v for k, v in train_metrics.items() if isinstance(v, (int, float))},
                **{f"holdout_{k}": v for k, v in holdout_metrics.items() if isinstance(v, (int, float))},
            })

        # Save model
        logger.info(f"\nSaving model to {output}")
        output.mkdir(parents=True, exist_ok=True)
        pipeline.save(str(output))

        # Save metrics
        metrics_file = output / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump({
                "train": train_metrics,
                "holdout": holdout_metrics,
            }, f, indent=2, default=str)

        logger.info(f"Metrics saved to {metrics_file}")

        # Log model to MLflow
        if mlflow:
            mlflow.log_artifacts(str(output))
            logger.info("Model artifacts logged to MLflow")

        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Model saved to: {output}")
        logger.info(f"Holdout positioning error: {holdout_metrics['mean_positioning_error']:.2f}m")
        logger.info(f"Building accuracy: {holdout_metrics['building_accuracy']:.1%}")
        logger.info(f"Floor accuracy: {holdout_metrics['floor_accuracy']:.1%}")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        if mlflow:
            mlflow.log_param("status", "failed")
        raise

    finally:
        if mlflow:
            mlflow.end_run()


def _flatten_dict(d: dict, parent_key: str = "", sep: str = "_") -> dict:
    """Flatten nested dictionary for MLflow logging.

    Args:
        d: Dictionary to flatten
        parent_key: Parent key for recursion
        sep: Separator for keys

    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _log_metrics(metrics: dict, prefix: str = "") -> None:
    """Log metrics to console.

    Args:
        metrics: Dictionary of metrics
        prefix: Prefix for metric names
    """
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {prefix}{key}: {value:.4f}")
        elif isinstance(value, int):
            logger.info(f"  {prefix}{key}: {value}")


if __name__ == "__main__":
    main()
