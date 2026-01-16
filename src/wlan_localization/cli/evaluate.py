"""CLI for evaluating trained cascade pipeline."""

import json
from pathlib import Path

import click
import pandas as pd

from wlan_localization import CascadePipeline
from wlan_localization.data import load_ujiindoorloc
from wlan_localization.evaluation.metrics import evaluate_per_building_floor
from wlan_localization.utils.logger import get_logger

logger = get_logger(__name__)


@click.command()
@click.option(
    "--model",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to trained model directory",
)
@click.option(
    "--data-dir",
    type=click.Path(exists=True, path_type=Path),
    default="data/raw",
    help="Directory containing test data",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default="reports/metrics/evaluation.json",
    help="Output file for evaluation metrics",
)
@click.option(
    "--save-predictions",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional: Save predictions to CSV file",
)
@click.option(
    "--building-penalty",
    type=float,
    default=50.0,
    help="Penalty for building misclassification (meters)",
)
@click.option(
    "--floor-penalty",
    type=float,
    default=4.0,
    help="Penalty for floor misclassification (meters)",
)
@click.option(
    "--detailed/--no-detailed",
    default=True,
    help="Generate detailed per-building-floor metrics",
)
def main(
    model: Path,
    data_dir: Path,
    output: Path,
    save_predictions: Path,
    building_penalty: float,
    floor_penalty: float,
    detailed: bool,
) -> None:
    """Evaluate trained cascade pipeline on test data.

    Example:
        wlan-evaluate --model models/cascade_trained --output reports/evaluation.json
    """
    logger.info("=" * 80)
    logger.info("WLAN INDOOR LOCALIZATION - EVALUATION")
    logger.info("=" * 80)

    # Load model
    logger.info(f"Loading model from {model}")
    pipeline = CascadePipeline.load(str(model))

    # Load test data
    logger.info(f"Loading test data from {data_dir}")
    X_train, y_train, X_test, y_test = load_ujiindoorloc(
        data_dir=str(data_dir),
        test_size=0.1,
        random_state=42,
    )

    logger.info(f"Test samples: {len(X_test)}")
    logger.info(f"Features: {X_test.shape[1]}")

    # Make predictions
    logger.info("\nGenerating predictions...")
    y_pred = pipeline.predict(X_test)

    # Evaluate overall performance
    logger.info("\nEvaluating performance...")
    metrics = pipeline.evaluate(
        X_test,
        y_test,
        building_penalty=building_penalty,
        floor_penalty=floor_penalty,
    )

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Mean Positioning Error: {metrics['mean_positioning_error']:.2f}m")
    logger.info(f"  ├─ Euclidean Distance: {metrics['mean_euclidean_distance']:.2f}m")
    logger.info(f"  ├─ Building Penalty: {building_penalty}m × failures")
    logger.info(f"  └─ Floor Penalty: {floor_penalty}m × failures")
    logger.info("\nBuilding Classification:")
    logger.info(f"  └─ Accuracy: {metrics['building_accuracy']:.1%}")
    logger.info("\nFloor Classification:")
    logger.info(f"  └─ Overall Accuracy: {metrics['floor_accuracy']:.1%}")

    # Per-building floor accuracy
    for building in [0, 1, 2]:
        key = f"building_{building}_floor_accuracy"
        if key in metrics:
            logger.info(f"     ├─ Building {building}: {metrics[key]:.1%}")

    # Detailed per-building-floor metrics
    if detailed:
        logger.info("\n" + "-" * 80)
        logger.info("PER-BUILDING-FLOOR PERFORMANCE")
        logger.info("-" * 80)

        detailed_metrics = evaluate_per_building_floor(y_test, y_pred)

        for (building, floor), row in detailed_metrics.iterrows():
            logger.info(
                f"Building {building}, Floor {floor}: "
                f"RMSE={row['rmse']:.2f}m, "
                f"MAE={row['mae']:.2f}m "
                f"(n={row['n_samples']})"
            )

        # Add detailed metrics to results
        metrics["per_location"] = detailed_metrics.to_dict()

    # Save predictions if requested
    if save_predictions:
        save_predictions.parent.mkdir(parents=True, exist_ok=True)

        # Combine true and predicted values
        results_df = pd.concat([
            y_test.reset_index(drop=True),
            y_pred.reset_index(drop=True).add_prefix("pred_")
        ], axis=1)

        # Add error column
        results_df["euclidean_error"] = (
            (results_df["LATITUDE"] - results_df["pred_LATITUDE"]) ** 2 +
            (results_df["LONGITUDE"] - results_df["pred_LONGITUDE"]) ** 2
        ) ** 0.5

        results_df.to_csv(save_predictions, index=False)
        logger.info(f"\nPredictions saved to {save_predictions}")

    # Save metrics
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    logger.info(f"Metrics saved to {output}")

    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
