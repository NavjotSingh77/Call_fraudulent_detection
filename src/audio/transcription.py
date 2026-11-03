"""
Speech-to-text transcription using OpenAI Whisper.

Requirements:
- Multilingual: preserves original spoken language
- Never translates Hindi or other languages into English
- Caches transcripts to avoid reprocessing
- Stores detected language, full transcript, and timestamped segments
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """
    Holds the full transcription output for a single call audio file.

    Attributes:
        call_id: Unique identifier for the call.
        audio_path: Absolute path to the source audio file (as string).
        language: BCP-47 language tag detected by Whisper (e.g., "hi", "en").
        language_probability: Whisper's confidence in the detected language [0, 1].
        transcript: Full concatenated transcript text in the original language.
        segments: List of timestamped segment dicts, each containing:
                  {id, start, end, text, confidence}.
        model_used: Name of the Whisper model used (e.g., "base", "large-v2")
                    or "ncsu_provided" for pre-existing transcripts.
        cached: True if this result was loaded from a cache file.
        error: Error message string if transcription failed, else None.
    """

    call_id: str
    audio_path: str
    language: str
    language_probability: float
    transcript: str
    segments: List[Dict]
    model_used: str
    cached: bool = False
    error: Optional[str] = None


def get_transcript_cache_path(call_id: str, transcripts_dir: Path) -> Path:
    """
    Return the filesystem path for a call's cached transcript JSON file.

    Args:
        call_id: Unique identifier for the call.
        transcripts_dir: Directory where transcript cache files are stored.

    Returns:
        Path object pointing to ``transcripts_dir/{call_id}_transcript.json``.
    """
    return transcripts_dir / f"{call_id}_transcript.json"


def load_cached_transcript(
    call_id: str,
    transcripts_dir: Path,
) -> Optional[TranscriptionResult]:
    """
    Load a previously cached transcription result from disk.

    Args:
        call_id: Unique identifier for the call.
        transcripts_dir: Directory where transcript cache files are stored.

    Returns:
        TranscriptionResult with ``cached=True`` if a valid cache file exists,
        or None if no cache is found or the file is malformed.
    """
    cache_path = get_transcript_cache_path(call_id, transcripts_dir)
    if not cache_path.exists():
        logger.debug("No cached transcript for call '%s'.", call_id)
        return None

    try:
        with cache_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        result = TranscriptionResult(
            call_id=data["call_id"],
            audio_path=data["audio_path"],
            language=data["language"],
            language_probability=float(data.get("language_probability", 0.0)),
            transcript=data["transcript"],
            segments=data.get("segments", []),
            model_used=data.get("model_used", "unknown"),
            cached=True,
            error=data.get("error"),
        )
        logger.info("Loaded cached transcript for call '%s'.", call_id)
        return result
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.warning(
            "Failed to load transcript cache for call '%s' from %s: %s",
            call_id,
            cache_path,
            exc,
        )
        return None


def save_transcript_cache(
    result: TranscriptionResult,
    transcripts_dir: Path,
) -> Path:
    """
    Persist a TranscriptionResult to a JSON cache file.

    Creates ``transcripts_dir`` if it does not already exist. Existing
    cache files for the same call_id are overwritten.

    Args:
        result: The TranscriptionResult to serialise and cache.
        transcripts_dir: Directory where transcript cache files are stored.

    Returns:
        Path to the written JSON file.
    """
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    cache_path = get_transcript_cache_path(result.call_id, transcripts_dir)

    payload = {
        "call_id": result.call_id,
        "audio_path": result.audio_path,
        "language": result.language,
        "language_probability": result.language_probability,
        "transcript": result.transcript,
        "segments": result.segments,
        "model_used": result.model_used,
        "cached": result.cached,
        "error": result.error,
    }

    try:
        with cache_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        logger.info("Saved transcript cache: %s", cache_path)
    except Exception as exc:
        logger.error(
            "Failed to save transcript cache for call '%s' to %s: %s",
            result.call_id,
            cache_path,
            exc,
        )
        raise

    return cache_path


def transcribe_audio(
    audio_path: Path,
    call_id: str,
    config: Dict,
    transcripts_dir: Path,
    force: bool = False,
) -> TranscriptionResult:
    """
    Transcribe an audio file using OpenAI Whisper, with caching.

    The function always uses ``task="transcribe"`` (NOT ``task="translate"``)
    so that Hindi, English, and other languages are preserved verbatim in the
    transcript. It never translates non-English audio into English.

    Processing order:
        1. If ``force=False``, check disk cache; return cached result if found.
        2. Attempt to import the ``whisper`` package. Return an error result
           if it is not installed.
        3. Load the Whisper model specified by ``config["whisper_model"]``
           (default: "base").
        4. Load the audio file; return an error result on failure.
        5. Run ``model.transcribe(audio, task="transcribe")``.
        6. Extract language, language_probability, transcript text, and
           per-segment timestamps with confidence scores.
        7. Save the result to the cache and return it.

    Args:
        audio_path: Path to the audio file to transcribe.
        call_id: Unique identifier for the call.
        config: Configuration dictionary. Key ``"whisper_model"`` sets the
                model name (e.g., "tiny", "base", "small", "medium", "large").
        transcripts_dir: Directory for caching transcript JSON files.
        force: If True, skip cache lookup and always re-transcribe.

    Returns:
        TranscriptionResult containing the transcript, language, segments,
        and metadata. On failure, ``error`` field is populated and
        ``transcript`` is an empty string.
    """
    # Step 1: Cache check
    if not force:
        cached = load_cached_transcript(call_id, transcripts_dir)
        if cached is not None:
            return cached

    model_name: str = config.get("whisper_model", "base")

    # Step 2: Check Whisper is available
    try:
        import whisper  # type: ignore
    except ImportError:
        err = (
            "OpenAI Whisper is not installed. "
            "Install it with: pip install openai-whisper"
        )
        logger.error(err)
        result = TranscriptionResult(
            call_id=call_id,
            audio_path=str(audio_path),
            language="",
            language_probability=0.0,
            transcript="",
            segments=[],
            model_used=model_name,
            cached=False,
            error=err,
        )
        return result

    # Step 3: Load the Whisper model
    try:
        logger.info("Loading Whisper model '%s'...", model_name)
        model = whisper.load_model(model_name)
        logger.info("Whisper model '%s' loaded.", model_name)
    except Exception as exc:
        err = f"Failed to load Whisper model '{model_name}': {exc}"
        logger.error(err)
        result = TranscriptionResult(
            call_id=call_id,
            audio_path=str(audio_path),
            language="",
            language_probability=0.0,
            transcript="",
            segments=[],
            model_used=model_name,
            cached=False,
            error=err,
        )
        return result

    # Step 4: Load audio
    if not audio_path.exists():
        err = f"Audio file not found: {audio_path}"
        logger.error(err)
        result = TranscriptionResult(
            call_id=call_id,
            audio_path=str(audio_path),
            language="",
            language_probability=0.0,
            transcript="",
            segments=[],
            model_used=model_name,
            cached=False,
            error=err,
        )
        return result

    try:
        audio_array = whisper.load_audio(str(audio_path))
    except Exception as exc:
        err = f"Failed to load audio for Whisper ('{audio_path}'): {exc}"
        logger.error(err)
        result = TranscriptionResult(
            call_id=call_id,
            audio_path=str(audio_path),
            language="",
            language_probability=0.0,
            transcript="",
            segments=[],
            model_used=model_name,
            cached=False,
            error=err,
        )
        return result

    # Step 5 & 6: Transcribe — task="transcribe" preserves original language
    try:
        logger.info(
            "Transcribing call '%s' with Whisper model '%s'...", call_id, model_name
        )
        whisper_result = model.transcribe(
            audio_array,
            task="transcribe",   # NEVER "translate" — preserve original language
            verbose=False,
        )

        detected_language: str = whisper_result.get("language", "")
        full_text: str = whisper_result.get("text", "").strip()

        # Extract language probability from the first segment's tokens if available
        # Whisper surfaces this in the result dict under some versions
        language_probability: float = float(
            whisper_result.get("language_probability", 0.0)
        )

        # Build per-segment list with confidence approximation
        raw_segments = whisper_result.get("segments", [])
        segments: List[Dict] = []
        for seg in raw_segments:
            # avg_logprob is Whisper's per-segment log-probability; convert to [0,1]
            avg_logprob: float = float(seg.get("avg_logprob", -1.0))
            # Clamp to a reasonable range and map to probability
            confidence: float = float(np.exp(max(avg_logprob, -5.0)))
            segments.append(
                {
                    "id": int(seg.get("id", len(segments))),
                    "start": float(seg.get("start", 0.0)),
                    "end": float(seg.get("end", 0.0)),
                    "text": str(seg.get("text", "")).strip(),
                    "confidence": round(confidence, 4),
                }
            )

        logger.info(
            "Transcription complete for call '%s': language='%s', %d segments, %d chars.",
            call_id,
            detected_language,
            len(segments),
            len(full_text),
        )

    except Exception as exc:
        err = f"Whisper transcription failed for call '{call_id}': {exc}"
        logger.error(err)
        result = TranscriptionResult(
            call_id=call_id,
            audio_path=str(audio_path),
            language="",
            language_probability=0.0,
            transcript="",
            segments=[],
            model_used=model_name,
            cached=False,
            error=err,
        )
        return result

    # Step 7: Build result and cache it
    result = TranscriptionResult(
        call_id=call_id,
        audio_path=str(audio_path),
        language=detected_language,
        language_probability=language_probability,
        transcript=full_text,
        segments=segments,
        model_used=model_name,
        cached=False,
        error=None,
    )

    try:
        save_transcript_cache(result, transcripts_dir)
    except Exception as exc:
        logger.warning("Could not save transcript cache for call '%s': %s", call_id, exc)

    return result


def transcribe_batch(
    audio_paths: List[Path],
    call_ids: List[str],
    config: Dict,
    transcripts_dir: Path,
    force: bool = False,
) -> List[TranscriptionResult]:
    """
    Transcribe multiple audio files sequentially with a progress bar.

    Continues processing remaining files even if individual transcriptions
    fail; failed items have their ``error`` field populated.

    Args:
        audio_paths: List of paths to audio files (must match call_ids length).
        call_ids: List of call identifiers corresponding to each audio file.
        config: Configuration dictionary (see transcribe_audio).
        transcripts_dir: Directory for caching transcript JSON files.
        force: If True, skip cache and re-transcribe all files.

    Returns:
        List of TranscriptionResult objects, one per input pair, in order.

    Raises:
        ValueError: If ``audio_paths`` and ``call_ids`` have different lengths.
    """
    if len(audio_paths) != len(call_ids):
        raise ValueError(
            f"audio_paths ({len(audio_paths)}) and call_ids ({len(call_ids)}) "
            "must have the same length."
        )

    try:
        from tqdm import tqdm  # type: ignore
        pairs = list(zip(audio_paths, call_ids))
        iterator = tqdm(pairs, desc="Transcribing calls", unit="file")
    except ImportError:
        logger.warning("tqdm not installed; progress bar disabled.")
        iterator = list(zip(audio_paths, call_ids))  # type: ignore

    results: List[TranscriptionResult] = []
    for audio_path, call_id in iterator:
        try:
            res = transcribe_audio(
                audio_path=audio_path,
                call_id=call_id,
                config=config,
                transcripts_dir=transcripts_dir,
                force=force,
            )
        except Exception as exc:
            logger.error(
                "Unexpected error transcribing call '%s' (%s): %s",
                call_id,
                audio_path,
                exc,
            )
            res = TranscriptionResult(
                call_id=call_id,
                audio_path=str(audio_path),
                language="",
                language_probability=0.0,
                transcript="",
                segments=[],
                model_used=config.get("whisper_model", "base"),
                cached=False,
                error=f"Unexpected error: {exc}",
            )
        results.append(res)

    success_count = sum(1 for r in results if r.error is None)
    logger.info(
        "Batch transcription complete: %d/%d files succeeded.",
        success_count,
        len(results),
    )
    return results


def load_existing_transcript(
    transcript_text: str,
    call_id: str,
    audio_path: str,
    language: str,
) -> TranscriptionResult:
    """
    Wrap a pre-existing transcript string (e.g., from NCSU metadata.csv) into
    a TranscriptionResult for uniform downstream handling.

    The NCSU dataset provides call-level transcripts without timestamps, so
    the ``segments`` list is left empty. The result is marked as cached with
    ``model_used="ncsu_provided"``.

    Args:
        transcript_text: The full transcript string from the NCSU metadata.
        call_id: Unique identifier for the call.
        audio_path: Path to the corresponding audio file (as string).
        language: Language of the transcript (e.g., "hi", "en").

    Returns:
        TranscriptionResult with ``cached=True``, ``model_used="ncsu_provided"``,
        ``language_probability=1.0`` (ground-truth label), and empty ``segments``.
    """
    return TranscriptionResult(
        call_id=call_id,
        audio_path=audio_path,
        language=language,
        language_probability=1.0,   # Provided by dataset — treat as certain
        transcript=transcript_text.strip(),
        segments=[],                # NCSU metadata does not include timestamps
        model_used="ncsu_provided",
        cached=True,
        error=None,
    )
