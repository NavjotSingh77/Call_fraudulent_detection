"""
Structured logging configuration for the Fraud Call Detection system.

Sets up console and optional file logging with color support.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "fraud_detection",
    level: str = "INFO",
    log_file: Optional[Path] = None,
    use_color: bool = True,
) -> logging.Logger:
    """Configure and return a named logger.

    Args:
        name: Logger name (typically module name).
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path to write log output to a file.
        use_color: Whether to use colored console output (requires colorlog).

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # Console handler
    try:
        if use_color:
            import colorlog
            color_fmt = (
                "%(log_color)s%(asctime)s | %(levelname)-8s%(reset)s | "
                "%(name)s | %(message)s"
            )
            console_handler = colorlog.StreamHandler(sys.stdout)
            console_handler.setFormatter(
                colorlog.ColoredFormatter(
                    color_fmt,
                    datefmt=datefmt,
                    log_colors={
                        "DEBUG": "cyan",
                        "INFO": "green",
                        "WARNING": "yellow",
                        "ERROR": "red",
                        "CRITICAL": "red,bg_white",
                    },
                )
            )
        else:
            raise ImportError
    except ImportError:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger by name. Uses root fraud_detection logger settings."""
    return logging.getLogger(f"fraud_detection.{name}")
