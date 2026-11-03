"""
Audio segmentation module for the Fraud Call Detection system.

Divides audio into overlapping windows for segment-level analysis.
Segment-level predictions are marked exploratory unless segment-level
annotations are available.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# Absolute minimum segment duration (seconds) — shorter segments are discarded.
MIN_SEGMENT_DURATION: float = 2.0


@dataclass
class AudioSegment:
    """
    Represents a single time-bounded segment of an audio call.

    Attributes:
        segment_id: Unique identifier for this segment (e.g., "call001_seg000").
        call_id: Identifier of the parent call this segment belongs to.
        start_time: Segment start time within the original audio, in seconds.
        end_time: Segment end time within the original audio, in seconds.
        duration: Segment duration in seconds (end_time - start_time).
        audio: Raw audio samples as a 1-D float32 numpy array.
        sample_rate: Sample rate of the audio data, in Hz.
        has_label: Whether this segment has a ground-truth label. Default False.
        label: Ground-truth label string if has_label is True. Default None.
        annotation_type: Annotation provenance. Default "exploratory" because
                         segment-level ground truth is generally unavailable;
                         call-level labels from NCSU metadata apply to the
                         whole call, not individual segments.
    """

    segment_id: str
    call_id: str
    start_time: float
    end_time: float
    duration: float
    audio: np.ndarray
    sample_rate: int
    has_label: bool = False
    label: Optional[str] = None
    annotation_type: str = "exploratory"


def get_segment_timestamps(
    audio_duration: float,
    segment_duration: float,
    stride: float,
) -> List[Tuple[float, float]]:
    """
    Calculate segment start/end timestamps for a given audio duration.

    Generates overlapping windows of ``segment_duration`` seconds, spaced
    ``stride`` seconds apart, without loading any audio data.

    The last window is extended to the end of the audio even if it would
    be shorter than ``segment_duration``, but windows shorter than
    MIN_SEGMENT_DURATION seconds are discarded.

    Args:
        audio_duration: Total duration of the audio in seconds.
        segment_duration: Desired length of each segment in seconds.
        stride: Step size between consecutive segment start times, in seconds.

    Returns:
        List of (start_sec, end_sec) tuples in chronological order.
    """
    if segment_duration <= 0:
        raise ValueError(f"segment_duration must be positive, got {segment_duration}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if audio_duration <= 0:
        return []

    timestamps: List[Tuple[float, float]] = []
    start = 0.0

    while start < audio_duration:
        end = min(start + segment_duration, audio_duration)
        duration = end - start
        if duration >= MIN_SEGMENT_DURATION:
            timestamps.append((start, end))
        start += stride

    logger.debug(
        "get_segment_timestamps: audio_duration=%.2fs, segment=%.1fs, stride=%.1fs -> %d segments",
        audio_duration,
        segment_duration,
        stride,
        len(timestamps),
    )
    return timestamps


def segment_audio(
    audio: np.ndarray,
    sr: int,
    call_id: str,
    segment_duration: float = 30.0,
    stride: float = 15.0,
) -> List[AudioSegment]:
    """
    Divide a full audio array into overlapping AudioSegment objects.

    Creates segments of ``segment_duration`` seconds with ``stride`` seconds
    between start times. The final segment may be shorter than
    ``segment_duration`` but must be at least MIN_SEGMENT_DURATION seconds
    long to be included. All segments are annotated as "exploratory" because
    segment-level ground-truth labels are not available in the NCSU dataset.

    Args:
        audio: 1-D float32 numpy array of audio samples.
        sr: Sample rate in Hz.
        call_id: Identifier of the parent call (used in segment IDs).
        segment_duration: Duration of each segment in seconds (default: 30.0).
        stride: Step size between consecutive segment start times in seconds
                (default: 15.0 — 50 % overlap with 30 s segments).

    Returns:
        Ordered list of AudioSegment objects.
    """
    if audio.ndim != 1:
        raise ValueError(
            f"segment_audio expects a 1-D audio array, got shape {audio.shape}"
        )

    audio_duration = float(len(audio)) / sr
    timestamps = get_segment_timestamps(audio_duration, segment_duration, stride)

    segments: List[AudioSegment] = []
    for idx, (start_sec, end_sec) in enumerate(timestamps):
        start_sample = int(start_sec * sr)
        end_sample = min(int(end_sec * sr), len(audio))
        segment_audio_data = audio[start_sample:end_sample].copy()

        actual_duration = float(len(segment_audio_data)) / sr
        segment_id = f"{call_id}_seg{idx:03d}"

        seg = AudioSegment(
            segment_id=segment_id,
            call_id=call_id,
            start_time=start_sec,
            end_time=end_sec,
            duration=actual_duration,
            audio=segment_audio_data,
            sample_rate=sr,
            has_label=False,
            label=None,
            annotation_type="exploratory",
        )
        segments.append(seg)

    logger.info(
        "Segmented call '%s': %d segments from %.2f s audio (seg=%.1fs, stride=%.1fs).",
        call_id,
        len(segments),
        audio_duration,
        segment_duration,
        stride,
    )
    return segments


def save_segments(
    segments: List[AudioSegment],
    output_dir: Path,
) -> List[Path]:
    """
    Save AudioSegment objects to WAV files in ``output_dir``.

    File naming convention:
        {call_id}_seg{idx:03d}_{start:.0f}s_{end:.0f}s.wav

    Args:
        segments: List of AudioSegment objects to save.
        output_dir: Directory in which to write the WAV files (created if
                    it does not exist).

    Returns:
        List of absolute Paths to the saved WAV files, in segment order.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []

    for idx, seg in enumerate(segments):
        filename = (
            f"{seg.call_id}_seg{idx:03d}_{seg.start_time:.0f}s_{seg.end_time:.0f}s.wav"
        )
        output_path = output_dir / filename

        try:
            sf.write(
                str(output_path),
                seg.audio,
                seg.sample_rate,
                subtype="PCM_16",
            )
            saved_paths.append(output_path)
            logger.debug("Saved segment: %s", output_path.name)
        except Exception as exc:
            logger.error(
                "Failed to save segment %s to %s: %s",
                seg.segment_id,
                output_path,
                exc,
            )

    logger.info(
        "Saved %d/%d segments for call '%s' to %s.",
        len(saved_paths),
        len(segments),
        segments[0].call_id if segments else "?",
        output_dir,
    )
    return saved_paths


def load_audio_for_segmentation(
    audio_path: Path,
    sr: int = 16000,
) -> Tuple[np.ndarray, int]:
    """
    Load a full audio file in preparation for segmentation.

    Loads the file as mono float32 at the specified sample rate using
    soundfile (with librosa fallback for formats soundfile cannot handle).

    Args:
        audio_path: Path to the audio file to load.
        sr: Target sample rate in Hz (default: 16000).

    Returns:
        Tuple of (audio_array, sample_rate) where audio_array is a 1-D
        float32 numpy array.

    Raises:
        FileNotFoundError: If ``audio_path`` does not exist.
        RuntimeError: If the file cannot be loaded by any available backend.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Primary: try soundfile (fast, low-memory)
    try:
        data, file_sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        # Mix down to mono if multichannel
        if data.ndim == 2:
            data = np.mean(data, axis=1)
        # Resample if needed
        if file_sr != sr:
            import librosa  # type: ignore
            data = librosa.resample(data, orig_sr=file_sr, target_sr=sr)
        logger.debug(
            "Loaded for segmentation: %s | samples=%d, sr=%d",
            audio_path.name,
            len(data),
            sr,
        )
        return data.astype(np.float32), sr
    except Exception as sf_exc:
        logger.warning(
            "soundfile failed for %s (%s); falling back to librosa.",
            audio_path.name,
            sf_exc,
        )

    # Fallback: librosa handles more formats (mp3, m4a, etc.)
    try:
        import librosa  # type: ignore
        data, _ = librosa.load(str(audio_path), sr=sr, mono=True, dtype=np.float32)
        logger.debug(
            "librosa fallback loaded: %s | samples=%d, sr=%d",
            audio_path.name,
            len(data),
            sr,
        )
        return data, sr
    except Exception as lr_exc:
        raise RuntimeError(
            f"Cannot load audio file '{audio_path}' with soundfile ({sf_exc}) "
            f"or librosa ({lr_exc})."
        ) from lr_exc
