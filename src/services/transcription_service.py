"""
Speech-to-text service leveraging Whisper and transcript caching.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from src.audio.transcription import (
    load_cached_transcript,
    save_transcript_cache,
    transcribe_audio,
)
from src.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class TranscriptionService:
    """Manages Whisper transcription with filesystem caching."""

    @staticmethod
    def transcribe(
        audio_path: Path,
        call_id: str,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Transcribe an audio file and return transcript, language, and segments."""
        transcripts_dir = settings.project_root / "data" / "processed" / "transcripts"
        transcripts_dir.mkdir(parents=True, exist_ok=True)

        # 1. Check cache first
        if not force:
            cached = load_cached_transcript(call_id, transcripts_dir)
            if cached and cached.transcript:
                logger.info("Retrieved transcript for %s from cache", call_id)
                return {
                    "call_id": cached.call_id,
                    "language": cached.language,
                    "transcript": cached.transcript,
                    "segments": cached.segments,
                    "model_used": cached.model_used,
                    "cached": True,
                }

        # 2. Transcribe using Whisper or fallback if Whisper is not installed
        config_dict = settings.load_yaml_config()
        # Override with environment settings if specified
        config_dict.setdefault("transcription", {})["whisper_model"] = settings.whisper_model
        config_dict["transcription"]["device"] = settings.device

        try:
            res = transcribe_audio(
                audio_path=audio_path,
                call_id=call_id,
                config=config_dict,
                transcripts_dir=transcripts_dir,
                force=force,
            )
            return {
                "call_id": res.call_id,
                "language": res.language or "en",
                "transcript": res.transcript,
                "segments": res.segments,
                "model_used": res.model_used,
                "cached": res.cached,
            }
        except Exception as exc:
            logger.warning("Transcription failed or Whisper unavailable: %s. Using fallback.", exc)
            return {
                "call_id": call_id,
                "language": "en",
                "transcript": "Audio received for fraud analysis.",
                "segments": [],
                "model_used": "fallback-offline",
                "cached": False,
            }
