"""
Application settings and environment configuration.

Loads configuration from environment variables (.env) and falls back
to config.yaml values for ML hyperparameters.
"""

from pathlib import Path
from typing import List, Optional
import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Centralized typed application settings."""

    # Project metadata
    app_name: str = "AI-Based Fraudulent Phone Call Detection"
    app_version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    app_env: str = Field(default="production", alias="APP_ENV")
    debug: bool = Field(default=False, alias="APP_DEBUG")
    log_level: str = Field(default="INFO", alias="APP_LOG_LEVEL")

    # Network / Gateway
    host: str = Field(default="0.0.0.0", alias="APP_HOST")
    port: int = Field(default=8000, alias="APP_PORT")
    cors_origins: List[str] = Field(default_factory=lambda: ["*"], alias="APP_CORS_ORIGINS")

    # Security (API key support for Android client)
    api_key_enabled: bool = Field(default=False, alias="APP_API_KEY_ENABLED")
    api_key: Optional[str] = Field(default=None, alias="APP_API_KEY")

    # Uploads
    max_upload_size_mb: int = Field(default=100, alias="APP_MAX_UPLOAD_SIZE_MB")

    # Compute & Hardware
    device: str = Field(default="cpu", alias="APP_DEVICE")

    # Speech Recognition
    whisper_model: str = Field(default="base", alias="APP_WHISPER_MODEL")

    # Database
    database_url: str = Field(default="sqlite:///./data/fraud_calls.db", alias="DATABASE_URL")

    # Model & Data Paths
    project_root: Path = Field(default_factory=_get_project_root)
    model_dir: Path = Field(default_factory=lambda: _get_project_root() / "models")

    # Disclaimer
    disclaimer: str = (
        "This prediction is a probabilistic model estimate and does NOT constitute "
        "legal or evidentiary proof of fraud. Human review is recommended for consequential decisions."
    )

    model_config = SettingsConfigDict(
        env_file=str(_get_project_root() / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def load_yaml_config(self) -> dict:
        """Load the project config.yaml if present."""
        cfg_path = self.project_root / "config.yaml"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}


_settings_instance: Optional[Settings] = None


def get_settings() -> Settings:
    """Retrieve singleton application settings instance."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance
