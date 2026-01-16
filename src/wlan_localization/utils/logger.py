"""Logging configuration using loguru."""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def get_logger(name: Optional[str] = None):
    """Get a configured logger instance.

    Args:
        name: Optional logger name (typically __name__ from calling module)

    Returns:
        Logger instance configured with appropriate handlers

    Example:
        >>> from wlan_localization.utils.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Training started")
    """
    # Remove default handler
    logger.remove()

    # Add stdout handler with INFO level
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    logger.add(
        sys.stdout,
        format=log_format,
        level="INFO",
        colorize=True
    )

    # Add file handler for persistent logs
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger.add(
        log_dir / "wlan_localization_{time}.log",
        rotation="500 MB",
        retention="10 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )

    if name:
        return logger.bind(name=name)

    return logger
