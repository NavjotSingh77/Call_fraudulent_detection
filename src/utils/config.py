"""
Configuration loader for the Fraud Call Detection system.

Loads config.yaml and provides typed access to all configuration values.
Supports overriding via environment variables.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to config.yaml. Defaults to project root config.yaml.

    Returns:
        Dictionary of configuration values.

    Raises:
        FileNotFoundError: If config file does not exist.
        yaml.YAMLError: If config file is malformed.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.debug(f"Loaded configuration from {path}")
    return config


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return _DEFAULT_CONFIG_PATH.parent


def resolve_path(config: Dict[str, Any], key_path: str) -> Path:
    """Resolve a relative path from config to an absolute project path.

    Args:
        config: Loaded configuration dictionary.
        key_path: Dot-separated key path, e.g. 'paths.data_raw'.

    Returns:
        Absolute Path object.
    """
    keys = key_path.split(".")
    value = config
    for k in keys:
        value = value[k]
    rel_path = Path(value)
    if rel_path.is_absolute():
        return rel_path
    return get_project_root() / rel_path


class Config:
    """Typed configuration wrapper for the Fraud Call Detection system."""

    def __init__(self, config_path: Optional[Path] = None):
        self._raw = load_config(config_path)
        self.project_root = get_project_root()

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get a config value by dot-separated key path."""
        keys = key_path.split(".")
        value = self._raw
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def path(self, key_path: str) -> Path:
        """Get an absolute path from a dot-separated key path."""
        return resolve_path(self._raw, key_path)

    def ensure_dirs(self) -> None:
        """Create all configured output directories if they do not exist."""
        path_keys = [
            "paths.data_raw",
            "paths.data_external",
            "paths.data_processed_audio",
            "paths.data_processed_transcripts",
            "paths.data_metadata",
            "paths.data_splits",
            "paths.models",
            "paths.figures",
            "paths.evaluation",
        ]
        for key in path_keys:
            p = self.path(key)
            p.mkdir(parents=True, exist_ok=True)
        # logs directory
        (self.project_root / "logs").mkdir(parents=True, exist_ok=True)

    @property
    def raw(self) -> Dict[str, Any]:
        """Return the raw configuration dictionary."""
        return self._raw
