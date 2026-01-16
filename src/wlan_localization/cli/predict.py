"""CLI for making predictions with trained cascade pipeline."""

import json
from pathlib import Path

import click
import numpy as np
import pandas as pd

from wlan_localization import CascadePipeline
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
    "--input",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Input CSV file with RSSI measurements",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    required=True,
    help="Output file for predictions (CSV or JSON)",
)
@click.option(
    "--format",
    type=click.Choice(["csv", "json"], case_sensitive=False),
    default="csv",
    help="Output format",
)
@click.option(
    "--batch-size",
    type=int,
    default=100,
    help="Batch size for processing large files",
)
def main(
    model: Path,
    input: Path,
    output: Path,
    format: str,
    batch_size: int,
) -> None:
    """Make predictions with trained cascade pipeline.

    Input CSV should contain WAP columns (RSSI values) matching the training data format.

    Example:
        wlan-predict --model models/cascade_trained \\
                     --input data/new_measurements.csv \\
                     --output predictions.csv
    """
    logger.info("=" * 80)
    logger.info("WLAN INDOOR LOCALIZATION - PREDICTION")
    logger.info("=" * 80)

    # Load model
    logger.info(f"Loading model from {model}")
    pipeline = CascadePipeline.load(str(model))

    # Load input data
    logger.info(f"Loading input data from {input}")
    data = pd.read_csv(input)

    # Validate input
    wap_columns = [col for col in data.columns if col.startswith("WAP")]
    if len(wap_columns) == 0:
        raise ValueError(
            "No WAP columns found in input file. "
            "Expected columns like WAP001, WAP002, etc."
        )

    logger.info(f"Found {len(wap_columns)} WAP columns")
    logger.info(f"Processing {len(data)} samples")

    # Extract features
    X = data[wap_columns].values

    # Make predictions
    logger.info("\nGenerating predictions...")

    if len(data) > batch_size:
        # Process in batches for large datasets
        predictions = []
        n_batches = int(np.ceil(len(data) / batch_size))

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(data))

            logger.info(f"  Batch {i+1}/{n_batches} ({start_idx}-{end_idx})")

            X_batch = X[start_idx:end_idx]
            pred_batch = pipeline.predict(X_batch)
            predictions.append(pred_batch)

        # Concatenate all predictions
        predictions_df = pd.concat(predictions, ignore_index=True)

    else:
        # Process all at once
        predictions_df = pipeline.predict(X)

    logger.info(f"Generated {len(predictions_df)} predictions")

    # Prepare output
    output.parent.mkdir(parents=True, exist_ok=True)

    if format.lower() == "csv":
        # Save as CSV
        # Optionally include input metadata
        if "TIMESTAMP" in data.columns or "USERID" in data.columns:
            metadata_cols = [col for col in ["TIMESTAMP", "USERID", "PHONEID"] if col in data.columns]
            result_df = pd.concat([
                data[metadata_cols].reset_index(drop=True),
                predictions_df.reset_index(drop=True)
            ], axis=1)
        else:
            result_df = predictions_df

        result_df.to_csv(output, index=False)
        logger.info(f"\nPredictions saved to {output}")

        # Print sample predictions
        logger.info("\nSample predictions:")
        logger.info(result_df.head(5).to_string())

    elif format.lower() == "json":
        # Save as JSON
        predictions_list = predictions_df.to_dict(orient="records")

        with open(output, "w") as f:
            json.dump(predictions_list, f, indent=2)

        logger.info(f"\nPredictions saved to {output}")

        # Print sample predictions
        logger.info("\nSample predictions:")
        for i, pred in enumerate(predictions_list[:5]):
            logger.info(f"  Sample {i+1}: Building {pred['BUILDINGID']}, "
                       f"Floor {pred['FLOOR']}, "
                       f"Location ({pred['LATITUDE']:.2f}, {pred['LONGITUDE']:.2f})")

    logger.info("\n" + "=" * 80)
    logger.info("PREDICTION COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
