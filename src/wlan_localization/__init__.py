"""WLAN Indoor Localization using Machine Learning.

A production-ready machine learning system for Wi-Fi fingerprint-based indoor
positioning using a cascaded architecture.
"""

__version__ = "2.0.0"
__author__ = "Sharan Naribole"
__license__ = "MIT"

from wlan_localization.models.cascade import CascadePipeline

__all__ = ["CascadePipeline", "__version__"]
