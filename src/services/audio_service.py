"""
Audio processing service for validating, storing, and preprocessing audio uploads.
"""

import logging
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Optional, Tuple

import soundfile as sf
from fastapi import HTTPException, UploadFile, status

from src.audio.preprocessing import (
    load_audio,
    normalize_amplitude,
    preprocess_audio,
    validate_audio_format,
)
from src.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


class AudioService:
    """Manages audio file validation, temporary persistence, and preprocessing."""

    @staticmethod
    def validate_file_metadata(file: UploadFile) -> str:
        """Validate file extension and size before reading."""
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file has no filename.",
            )

        ext = Path(file.filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported audio format '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
            )
        return ext

    @staticmethod
    async def save_upload_to_temp(file: UploadFile) -> Path:
        """Asynchronously stream upload contents into a secure temporary file."""
        AudioService.validate_file_metadata(file)
        ext = Path(file.filename or "audio.wav").suffix.lower()

        max_bytes = settings.max_upload_size_mb * 1024 * 1024
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        written_bytes = 0

        try:
            while chunk := await file.read(64 * 1024):
                written_bytes += len(chunk)
                if written_bytes > max_bytes:
                    temp_file.close()
                    Path(temp_file.name).unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"Audio file exceeds maximum size limit of {settings.max_upload_size_mb} MB.",
                    )
                temp_file.write(chunk)
            temp_file.flush()
        finally:
            temp_file.close()

        return Path(temp_file.name)

    @staticmethod
    def get_audio_info(path: Path) -> Dict:
        """Retrieve duration, sample rate, channels using soundfile."""
        fmt_res = validate_audio_format(path)
        if not fmt_res["valid"]:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Audio format validation failed: {fmt_res.get('error')}",
            )
        return fmt_res

    @staticmethod
    def preprocess_call_audio(raw_path: Path, output_dir: Optional[Path] = None) -> Path:
        """Resample to 16kHz mono and apply amplitude normalization."""
        if output_dir is None:
            output_dir = settings.project_root / "data" / "processed" / "audio"
        output_dir.mkdir(parents=True, exist_ok=True)

        config_dict = settings.load_yaml_config()
        res = preprocess_audio(raw_path, output_dir, config_dict)
        out_p = res.get("output_path") or res.get("processed_path")
        if not out_p:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Audio preprocessing failed: {res.get('error')}",
            )
        return Path(out_p)
