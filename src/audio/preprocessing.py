"""
Audio preprocessing module for the Fraud Call Detection system.

Handles audio validation, format conversion, resampling, normalization,
silence detection, and corrupt file detection.

Preserves original files; saves processed audio to a separate directory.
Avoids aggressive preprocessing that would remove meaningful speech.
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# Supported audio file extensions
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".wma"}


class AudioLoadError(Exception):
    """Raised when an audio file cannot be loaded."""
    pass


def validate_audio_format(audio_path: Path) -> Dict:
    """
    Validate an audio file's format and extract metadata.

    Checks file existence, supported extension, and attempts to open with
    soundfile to retrieve sample rate, channels, duration, and format.
    Catches corrupt or unreadable files gracefully.

    Args:
        audio_path: Path to the audio file to validate.

    Returns:
        Dictionary with keys:
            - valid (bool): Whether the file is valid and readable.
            - sample_rate (int): Sample rate in Hz, or 0 on failure.
            - channels (int): Number of audio channels, or 0 on failure.
            - duration_seconds (float): Duration in seconds, or 0.0 on failure.
            - format (str): File format string (e.g., "WAV"), or "" on failure.
            - error (str | None): Error message if invalid, else None.
    """
    result: Dict = {
        "valid": False,
        "sample_rate": 0,
        "channels": 0,
        "duration_seconds": 0.0,
        "format": "",
        "error": None,
    }

    # Check file existence
    if not audio_path.exists():
        result["error"] = f"File not found: {audio_path}"
        logger.warning(result["error"])
        return result

    if not audio_path.is_file():
        result["error"] = f"Path is not a file: {audio_path}"
        logger.warning(result["error"])
        return result

    # Check supported extension
    ext = audio_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        result["error"] = (
            f"Unsupported file extension '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
        logger.warning(result["error"])
        return result

    # Try opening with soundfile to get metadata
    try:
        info = sf.info(str(audio_path))
        result["valid"] = True
        result["sample_rate"] = info.samplerate
        result["channels"] = info.channels
        result["duration_seconds"] = info.duration
        result["format"] = info.format
        result["error"] = None
        logger.debug(
            "Audio validated: %s | sr=%d, ch=%d, dur=%.2fs, fmt=%s",
            audio_path.name,
            info.samplerate,
            info.channels,
            info.duration,
            info.format,
        )
    except Exception as exc:
        result["error"] = f"Cannot read audio file (possibly corrupt): {exc}"
        logger.warning("Validation failed for %s: %s", audio_path.name, exc)

    return result


def load_audio(
    audio_path: Path,
    target_sr: int = 16000,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file, optionally convert to mono, and resample.

    Uses librosa for robust format support and resampling. The returned
    array is always float32 in the range [-1.0, 1.0].

    Args:
        audio_path: Path to the audio file.
        target_sr: Desired sample rate in Hz (default: 16000).
        mono: If True, mix down to a single channel (default: True).

    Returns:
        Tuple of (audio_array, sample_rate) where audio_array is np.ndarray
        of shape (samples,) for mono or (channels, samples) for stereo.

    Raises:
        AudioLoadError: If the file cannot be loaded for any reason.
    """
    if not audio_path.exists():
        raise AudioLoadError(f"Audio file not found: {audio_path}")

    try:
        audio, sr = librosa.load(
            str(audio_path),
            sr=target_sr,
            mono=mono,
            dtype=np.float32,
        )
        logger.debug(
            "Loaded audio: %s | shape=%s, sr=%d",
            audio_path.name,
            audio.shape,
            sr,
        )
        return audio, sr
    except Exception as exc:
        raise AudioLoadError(
            f"Failed to load audio file '{audio_path}': {exc}"
        ) from exc


def normalize_amplitude(
    audio: np.ndarray,
    target_level: float = -3.0,
) -> np.ndarray:
    """
    Apply peak normalization to a target level in dBFS.

    Scales the audio so that the peak sample magnitude reaches the
    specified target level in decibels relative to full scale (dBFS).
    Silent audio (all zeros) is returned unchanged to avoid division by zero.

    Args:
        audio: Input audio array (float32, any shape).
        target_level: Target peak level in dBFS (default: -3.0).
                      Must be <= 0.0.

    Returns:
        Normalized audio array of the same shape and dtype as input.
    """
    peak = np.max(np.abs(audio))
    if peak < 1e-8:
        # Audio is silent — skip normalization to avoid divide-by-zero
        logger.debug("normalize_amplitude: silent audio detected, skipping normalization.")
        return audio.copy()

    target_linear = 10 ** (target_level / 20.0)
    gain = target_linear / peak
    normalized = audio * gain

    # Hard-clip to [-1, 1] as safety guard
    normalized = np.clip(normalized, -1.0, 1.0)
    logger.debug(
        "normalize_amplitude: peak=%.6f, gain=%.4f, target=%.1f dBFS",
        peak,
        gain,
        target_level,
    )
    return normalized.astype(audio.dtype)


def detect_silence(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = -60.0,
    min_silence_duration: float = 0.5,
) -> Dict:
    """
    Detect silent regions within an audio array.

    A frame is considered silent if its RMS energy falls below the
    threshold_db level. Consecutive silent frames are merged into regions
    only if they span at least min_silence_duration seconds.

    Args:
        audio: 1-D audio array (float32).
        sr: Sample rate of the audio.
        threshold_db: Energy threshold in dBFS below which a frame is silent
                      (default: -60.0).
        min_silence_duration: Minimum duration in seconds for a region to be
                              reported as a silent region (default: 0.5).

    Returns:
        Dictionary with keys:
            - is_mostly_silent (bool): True if silence_ratio > 0.9.
            - silence_ratio (float): Fraction of audio that is silent [0, 1].
            - silent_regions (List[Dict]): Each dict has 'start', 'end',
              'duration' in seconds.
    """
    frame_length = int(sr * 0.025)   # 25 ms frames
    hop_length = int(sr * 0.010)     # 10 ms hop

    # Compute per-frame RMS energy
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]

    # Convert threshold from dBFS to linear
    threshold_linear = 10 ** (threshold_db / 20.0)

    silence_mask = rms < threshold_linear  # shape: (n_frames,)

    total_frames = len(silence_mask)
    silent_frame_count = int(np.sum(silence_mask))
    silence_ratio = silent_frame_count / total_frames if total_frames > 0 else 1.0

    # Identify contiguous silent regions
    silent_regions: List[Dict] = []
    min_silent_frames = int(min_silence_duration / 0.010)  # hop = 10 ms

    in_silence = False
    region_start_frame = 0

    for i, is_silent in enumerate(silence_mask):
        if is_silent and not in_silence:
            in_silence = True
            region_start_frame = i
        elif not is_silent and in_silence:
            in_silence = False
            region_len = i - region_start_frame
            if region_len >= min_silent_frames:
                start_sec = librosa.frames_to_time(
                    region_start_frame, sr=sr, hop_length=hop_length
                )
                end_sec = librosa.frames_to_time(i, sr=sr, hop_length=hop_length)
                silent_regions.append(
                    {
                        "start": float(start_sec),
                        "end": float(end_sec),
                        "duration": float(end_sec - start_sec),
                    }
                )

    # Handle trailing silence
    if in_silence:
        region_len = total_frames - region_start_frame
        if region_len >= min_silent_frames:
            start_sec = librosa.frames_to_time(
                region_start_frame, sr=sr, hop_length=hop_length
            )
            end_sec = float(len(audio)) / sr
            silent_regions.append(
                {
                    "start": float(start_sec),
                    "end": end_sec,
                    "duration": float(end_sec - start_sec),
                }
            )

    is_mostly_silent = silence_ratio > 0.9

    logger.debug(
        "detect_silence: silence_ratio=%.3f, regions=%d, mostly_silent=%s",
        silence_ratio,
        len(silent_regions),
        is_mostly_silent,
    )

    return {
        "is_mostly_silent": is_mostly_silent,
        "silence_ratio": float(silence_ratio),
        "silent_regions": silent_regions,
    }


def detect_corrupt_audio(audio_path: Path) -> Dict:
    """
    Check if an audio file is corrupt or unreadable.

    Performs multiple checks:
    1. File can be opened and read by soundfile.
    2. Audio data contains no NaN or infinite values.
    3. Audio is not zero-length.

    Args:
        audio_path: Path to the audio file.

    Returns:
        Dictionary with keys:
            - is_corrupt (bool): True if the file is considered corrupt.
            - reason (str | None): Human-readable reason, or None if not corrupt.
    """
    if not audio_path.exists():
        return {"is_corrupt": True, "reason": f"File not found: {audio_path}"}

    # Attempt to read audio data with soundfile
    try:
        data, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    except Exception as exc:
        return {"is_corrupt": True, "reason": f"Cannot read file: {exc}"}

    # Check for zero-length audio
    if data.size == 0 or len(data) == 0:
        return {"is_corrupt": True, "reason": "Audio file contains no samples (zero length)."}

    # Check for NaN values
    if np.any(np.isnan(data)):
        return {"is_corrupt": True, "reason": "Audio data contains NaN values."}

    # Check for infinite values
    if np.any(np.isinf(data)):
        return {"is_corrupt": True, "reason": "Audio data contains infinite (Inf) values."}

    logger.debug("detect_corrupt_audio: %s is valid.", audio_path.name)
    return {"is_corrupt": False, "reason": None}


def preprocess_audio(
    audio_path: Path,
    output_dir: Path,
    config: Dict,
) -> Dict:
    """
    Full preprocessing pipeline for a single audio file.

    Steps:
        1. Validate format
        2. Check for corruption
        3. Load audio (librosa)
        4. Convert to mono
        5. Resample to 16 kHz
        6. Normalize amplitude
        7. Detect silence
        8. Optionally apply noise reduction (if config audio.noise_reduction=True
           and noisereduce package is installed)
        9. Save processed audio to output_dir/processed_{original_name}.wav
        10. Return a comprehensive result dict with all metadata

    Original files are never modified or overwritten.

    Args:
        audio_path: Path to the source audio file.
        output_dir: Directory where the processed WAV will be saved.
        config: Configuration dictionary. Relevant keys under "audio":
                - target_sr (int, default 16000)
                - target_level_db (float, default -3.0)
                - silence_threshold_db (float, default -60.0)
                - noise_reduction (bool, default False)

    Returns:
        Dictionary with comprehensive metadata and processing results,
        including 'success' (bool) and 'error' (str | None).
    """
    audio_config = config.get("audio", {})
    target_sr: int = int(audio_config.get("target_sr", 16000))
    target_level_db: float = float(audio_config.get("target_level_db", -3.0))
    silence_threshold_db: float = float(audio_config.get("silence_threshold_db", -60.0))
    apply_noise_reduction: bool = bool(audio_config.get("noise_reduction", False))

    result: Dict = {
        "audio_path": str(audio_path),
        "output_path": None,
        "success": False,
        "error": None,
        "validation": None,
        "corruption_check": None,
        "silence_info": None,
        "sample_rate": target_sr,
        "duration_seconds": 0.0,
        "channels": 1,
        "noise_reduction_applied": False,
    }

    # Step 1: Validate format
    logger.info("Preprocessing: %s", audio_path.name)
    validation = validate_audio_format(audio_path)
    result["validation"] = validation
    if not validation["valid"]:
        result["error"] = f"Validation failed: {validation['error']}"
        logger.error(result["error"])
        return result

    # Step 2: Check for corruption
    corruption = detect_corrupt_audio(audio_path)
    result["corruption_check"] = corruption
    if corruption["is_corrupt"]:
        result["error"] = f"Corrupt audio: {corruption['reason']}"
        logger.error(result["error"])
        return result

    # Step 3 & 4 & 5: Load, mono-mix, and resample
    try:
        audio, sr = load_audio(audio_path, target_sr=target_sr, mono=True)
    except AudioLoadError as exc:
        result["error"] = str(exc)
        logger.error(result["error"])
        return result

    result["duration_seconds"] = float(len(audio)) / sr
    result["sample_rate"] = sr
    result["channels"] = 1

    # Step 6: Normalize amplitude
    audio = normalize_amplitude(audio, target_level=target_level_db)

    # Step 7: Detect silence
    silence_info = detect_silence(audio, sr, threshold_db=silence_threshold_db)
    result["silence_info"] = silence_info
    if silence_info["is_mostly_silent"]:
        logger.warning(
            "Audio is mostly silent (ratio=%.2f): %s",
            silence_info["silence_ratio"],
            audio_path.name,
        )

    # Step 8: Optional noise reduction
    if apply_noise_reduction:
        try:
            import noisereduce as nr  # type: ignore
            audio = nr.reduce_noise(y=audio, sr=sr).astype(np.float32)
            result["noise_reduction_applied"] = True
            logger.debug("Noise reduction applied to %s", audio_path.name)
        except ImportError:
            logger.warning(
                "noisereduce package not installed; skipping noise reduction for %s",
                audio_path.name,
            )
        except Exception as exc:
            logger.warning(
                "Noise reduction failed for %s: %s — skipping.", audio_path.name, exc
            )

    # Step 9: Save processed audio (never overwrite originals)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = f"processed_{audio_path.stem}.wav"
    output_path = output_dir / output_filename

    try:
        sf.write(str(output_path), audio, sr, subtype="PCM_16")
        result["output_path"] = str(output_path)
        logger.info("Saved processed audio: %s", output_path)
    except Exception as exc:
        result["error"] = f"Failed to save processed audio: {exc}"
        logger.error(result["error"])
        return result

    result["success"] = True
    return result


def preprocess_batch(
    audio_paths: List[Path],
    output_dir: Path,
    config: Dict,
) -> List[Dict]:
    """
    Preprocess multiple audio files with a progress bar.

    Continues processing remaining files even if individual files fail.
    Each file's result includes a 'success' flag and 'error' details.

    Args:
        audio_paths: List of paths to audio files to process.
        output_dir: Directory where processed WAVs will be saved.
        config: Configuration dictionary (see preprocess_audio).

    Returns:
        List of result dicts, one per input file (in same order).
    """
    try:
        from tqdm import tqdm  # type: ignore
        iterator = tqdm(audio_paths, desc="Preprocessing audio", unit="file")
    except ImportError:
        logger.warning("tqdm not installed; progress bar disabled.")
        iterator = audio_paths  # type: ignore

    results: List[Dict] = []
    for audio_path in iterator:
        try:
            res = preprocess_audio(audio_path, output_dir, config)
        except Exception as exc:
            logger.error("Unexpected error processing %s: %s", audio_path, exc)
            res = {
                "audio_path": str(audio_path),
                "output_path": None,
                "success": False,
                "error": f"Unexpected error: {exc}",
                "validation": None,
                "corruption_check": None,
                "silence_info": None,
                "sample_rate": 0,
                "duration_seconds": 0.0,
                "channels": 0,
                "noise_reduction_applied": False,
            }
        results.append(res)

    success_count = sum(1 for r in results if r["success"])
    logger.info(
        "Batch preprocessing complete: %d/%d files succeeded.",
        success_count,
        len(results),
    )
    return results


def get_audio_duration(audio_path: Path) -> float:
    """
    Quickly retrieve the duration of an audio file without full load.

    Uses soundfile's info() for an efficient metadata-only read.

    Args:
        audio_path: Path to the audio file.

    Returns:
        Duration in seconds as a float.

    Raises:
        AudioLoadError: If the file cannot be read or duration is unavailable.
    """
    try:
        info = sf.info(str(audio_path))
        return float(info.duration)
    except Exception as exc:
        raise AudioLoadError(
            f"Cannot determine duration of '{audio_path}': {exc}"
        ) from exc
