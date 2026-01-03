"""Logging configuration module."""

import sys
from pathlib import Path
from loguru import logger
from src.config.settings import get_settings


def setup_logging(log_to_file: bool = True) -> None:
    """Configure application logging with loguru."""
    settings = get_settings()

    # Remove default handler
    logger.remove()

    # Console handler
    logger.add(
        sys.stdout,
        format=settings.logging.format,
        level=settings.logging.level,
        colorize=True,
    )

    # File handler
    if log_to_file and settings.logging.log_file:
        log_path = Path(settings.logging.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            str(log_path),
            format=settings.logging.format,
            level=settings.logging.level,
            rotation=settings.logging.rotation,
            retention=settings.logging.retention,
        )

    logger.info(f"Logging configured: level={settings.logging.level}")
