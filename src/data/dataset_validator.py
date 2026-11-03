"""
Dataset validation module.

Validates audio files, transcripts, and manifest integrity.
Produces a validation report with per-file status.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Default minimum acceptable audio duration (seconds)
_DEFAULT_MIN_DURATION: float = 0.5


# ---------------------------------------------------------------------------
# 1. validate_audio_file
# ---------------------------------------------------------------------------


def validate_audio_file(
    audio_path: Path,
    min_duration: float = _DEFAULT_MIN_DURATION,
) -> Dict[str, Any]:
    """Validate a single audio file.

    Checks:
    * The file exists on disk.
    * The file can be opened and read by *soundfile*.
    * The audio duration is at least *min_duration* seconds.

    Parameters
    ----------
    audio_path:
        Path to the audio file to validate.
    min_duration:
        Minimum required duration in seconds (default 0.5 s).

    Returns
    -------
    Dict
        Keys: ``valid`` (bool), ``duration`` (float|None),
        ``sample_rate`` (int|None), ``channels`` (int|None),
        ``error`` (str|None).
    """
    audio_path = Path(audio_path)
    result: Dict[str, Any] = {
        "valid": False,
        "duration": None,
        "sample_rate": None,
        "channels": None,
        "error": None,
    }

    if not audio_path.exists():
        result["error"] = f"File not found: {audio_path}"
        return result

    try:
        import soundfile as sf  # lazy import – not always installed

        info = sf.info(audio_path)
        duration: float = info.frames / info.samplerate
        result["duration"] = duration
        result["sample_rate"] = info.samplerate
        result["channels"] = info.channels

        if duration < min_duration:
            result["error"] = (
                f"Duration {duration:.3f}s is below minimum {min_duration}s."
            )
            return result

        result["valid"] = True
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"soundfile error: {exc}"

    return result


# ---------------------------------------------------------------------------
# 2. validate_manifest
# ---------------------------------------------------------------------------


def validate_manifest(
    manifest_df: pd.DataFrame,
    config: Dict,
) -> pd.DataFrame:
    """Validate audio files referenced in *manifest_df*.

    For every row that has a non-null ``audio_path``, the corresponding
    file is validated with :func:`validate_audio_file`.  New columns are
    appended to the returned copy of the DataFrame.

    Parameters
    ----------
    manifest_df:
        Manifest DataFrame (as produced by
        ``src.data.ingestion.build_manifest``).
    config:
        Project configuration dict.  May contain
        ``"min_audio_duration"`` (float, default 0.5).

    Returns
    -------
    pd.DataFrame
        Copy of *manifest_df* enriched with columns:
        ``audio_valid``, ``audio_duration``, ``audio_sample_rate``,
        ``validation_error``.
    """
    min_duration: float = config.get("min_audio_duration", _DEFAULT_MIN_DURATION)
    df = manifest_df.copy()

    audio_valid: list = []
    audio_duration: list = []
    audio_sample_rate: list = []
    validation_error: list = []

    for _, row in df.iterrows():
        path_val = row.get("audio_path")
        if pd.isna(path_val) or str(path_val).strip() == "":
            audio_valid.append(None)
            audio_duration.append(None)
            audio_sample_rate.append(None)
            validation_error.append("No audio_path specified.")
            continue

        vresult = validate_audio_file(Path(str(path_val)), min_duration=min_duration)
        audio_valid.append(vresult["valid"])
        audio_duration.append(vresult["duration"])
        audio_sample_rate.append(vresult["sample_rate"])
        validation_error.append(vresult["error"])

    df["audio_valid"] = audio_valid
    df["audio_duration"] = audio_duration
    df["audio_sample_rate"] = audio_sample_rate
    df["validation_error"] = validation_error

    valid_count = sum(1 for v in audio_valid if v is True)
    logger.info(
        "Audio validation complete: %d / %d files valid.",
        valid_count,
        len(df),
    )
    return df


# ---------------------------------------------------------------------------
# 3. validate_transcripts
# ---------------------------------------------------------------------------


def validate_transcripts(manifest_df: pd.DataFrame) -> pd.DataFrame:
    """Check that each record has a non-empty transcript.

    Parameters
    ----------
    manifest_df:
        Manifest DataFrame (must contain a ``transcript`` column).

    Returns
    -------
    pd.DataFrame
        Copy enriched with the boolean column ``transcript_available``.
    """
    df = manifest_df.copy()

    if "transcript" not in df.columns:
        logger.warning(
            "No 'transcript' column found in manifest. "
            "Marking all records as transcript_available=False."
        )
        df["transcript_available"] = False
        return df

    df["transcript_available"] = (
        df["transcript"].notna()
        & (df["transcript"].astype(str).str.strip() != "")
    )

    available_count = df["transcript_available"].sum()
    logger.info(
        "Transcript validation: %d / %d records have a non-empty transcript.",
        available_count,
        len(df),
    )
    return df


# ---------------------------------------------------------------------------
# 4. generate_validation_report
# ---------------------------------------------------------------------------


def generate_validation_report(
    manifest_df: pd.DataFrame,
    output_path: Path,
) -> Dict[str, Any]:
    """Generate a JSON validation summary report.

    Parameters
    ----------
    manifest_df:
        Manifest DataFrame that has been enriched by
        :func:`validate_manifest` and :func:`validate_transcripts`.
    output_path:
        File path where the JSON report will be written.

    Returns
    -------
    Dict
        Summary statistics dict (also written to *output_path*).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(manifest_df)

    # Audio stats
    audio_col_present = "audio_valid" in manifest_df.columns
    audio_valid_count: int = (
        int(manifest_df["audio_valid"].sum()) if audio_col_present else 0
    )
    audio_invalid_count: int = (
        int((manifest_df["audio_valid"] == False).sum())  # noqa: E712
        if audio_col_present
        else 0
    )
    audio_missing_count: int = (
        int(manifest_df["audio_valid"].isna().sum()) if audio_col_present else total
    )

    # Transcript stats
    transcript_col_present = "transcript_available" in manifest_df.columns
    transcript_available_count: int = (
        int(manifest_df["transcript_available"].sum())
        if transcript_col_present
        else 0
    )

    # Label stats
    label_dist: Dict[str, int] = (
        manifest_df["label"].value_counts().to_dict()
        if "label" in manifest_df.columns
        else {}
    )
    label_status_dist: Dict[str, int] = (
        manifest_df["label_status"].value_counts().to_dict()
        if "label_status" in manifest_df.columns
        else {}
    )

    # Collect error messages
    validation_errors: list = []
    if audio_col_present:
        error_rows = manifest_df[
            manifest_df["validation_error"].notna()
            & (manifest_df["validation_error"] != "No audio_path specified.")
        ]
        validation_errors = error_rows["validation_error"].tolist()

    report: Dict[str, Any] = {
        "total_records": total,
        "audio": {
            "valid": audio_valid_count,
            "invalid": audio_invalid_count,
            "missing": audio_missing_count,
        },
        "transcripts": {
            "available": transcript_available_count,
            "missing": total - transcript_available_count,
        },
        "label_distribution": label_dist,
        "label_status_distribution": label_status_dist,
        "validation_errors_sample": validation_errors[:20],
    }

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    logger.info("Validation report saved to %s", output_path)
    return report
