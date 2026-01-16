"""Command-line interface modules."""

from wlan_localization.cli.evaluate import main as evaluate_main
from wlan_localization.cli.predict import main as predict_main
from wlan_localization.cli.train import main as train_main

__all__ = ["train_main", "evaluate_main", "predict_main"]
