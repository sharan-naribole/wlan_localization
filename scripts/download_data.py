"""Script to download UJIIndoorLoc dataset from UCI ML Repository."""

import sys
from pathlib import Path

import requests
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wlan_localization.utils.logger import get_logger

logger = get_logger(__name__)

# UCI ML Repository URLs
TRAINING_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00310/UJIndoorLoc/trainingData.csv"
VALIDATION_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00310/UJIndoorLoc/validationData.csv"


def download_file(url: str, output_path: Path) -> None:
    """Download file with progress bar.

    Args:
        url: URL to download from
        output_path: Local path to save file
    """
    logger.info(f"Downloading {url}")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(output_path, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    logger.info(f"Saved to {output_path}")


def main() -> None:
    """Download UJIIndoorLoc dataset."""
    logger.info("=" * 80)
    logger.info("UJIIndoorLoc Dataset Download")
    logger.info("=" * 80)

    # Create data directory
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download training data
    training_file = data_dir / "trainingData.csv"
    if training_file.exists():
        logger.info(f"Training data already exists at {training_file}")
    else:
        try:
            download_file(TRAINING_DATA_URL, training_file)
        except Exception as e:
            logger.error(f"Failed to download training data: {e}")
            logger.info("\nManual download instructions:")
            logger.info("1. Visit: https://archive.ics.uci.edu/ml/datasets/UJIIndoorLoc")
            logger.info("2. Download 'trainingData.csv' and 'validationData.csv'")
            logger.info(f"3. Place them in {data_dir}")
            return

    # Download validation data
    validation_file = data_dir / "validationData.csv"
    if validation_file.exists():
        logger.info(f"Validation data already exists at {validation_file}")
    else:
        try:
            download_file(VALIDATION_DATA_URL, validation_file)
        except Exception as e:
            logger.error(f"Failed to download validation data: {e}")
            logger.info("\nManual download instructions:")
            logger.info("1. Visit: https://archive.ics.uci.edu/ml/datasets/UJIIndoorLoc")
            logger.info("2. Download 'validationData.csv'")
            logger.info(f"3. Place it in {data_dir}")
            return

    # Verify files
    logger.info("\n" + "=" * 80)
    logger.info("Dataset Download Complete")
    logger.info("=" * 80)
    logger.info(f"Training data: {training_file} ({training_file.stat().st_size / 1e6:.1f} MB)")
    logger.info(f"Validation data: {validation_file} ({validation_file.stat().st_size / 1e6:.1f} MB)")
    logger.info("\nYou can now train the model using:")
    logger.info("  python scripts/train.py")
    logger.info("or")
    logger.info("  wlan-train --config configs/cascade_optimal.yaml")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nDownload cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Download failed: {e}", exc_info=True)
        sys.exit(1)
