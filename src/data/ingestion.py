"""
Dataset ingestion module for the NCSU Robocall Audio Dataset.

Downloads metadata.csv, audio files, and PDF references from the NCSU
GitHub repository. Creates a standardized dataset manifest.

Dataset source: https://github.com/wspr-ncsu/robocall-audio-dataset
All recordings are from FTC 'Project Point of No Entry' enforcement actions.
Label: fraud, label_status: verified (FTC enforcement-backed evidence).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Repository constants
# ---------------------------------------------------------------------------

_GITHUB_RAW_BASE = (
    "https://raw.githubusercontent.com/wspr-ncsu/robocall-audio-dataset/main"
)
_GITHUB_TREE_API = (
    "https://api.github.com/repos/wspr-ncsu/robocall-audio-dataset"
    "/git/trees/main?recursive=1"
)
_METADATA_REMOTE_PATH = "metadata.csv"
_AUDIO_GITHUB_DIR = "audio"  # directory inside the repo that holds .wav files


# ---------------------------------------------------------------------------
# 1. download_file
# ---------------------------------------------------------------------------


def download_file(url: str, dest_path: Path, chunk_size: int = 8192) -> bool:
    """Download a file from *url* to *dest_path*, showing a progress bar.

    Parameters
    ----------
    url:
        Full URL of the remote resource.
    dest_path:
        Local destination path (parent directories must exist).
    chunk_size:
        Number of bytes to read per iteration.

    Returns
    -------
    bool
        ``True`` on success, ``False`` on any HTTP or I/O error.
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        logger.error("HTTP error downloading %s: %s", url, exc)
        return False
    except requests.exceptions.RequestException as exc:
        logger.error("Request error downloading %s: %s", url, exc)
        return False

    total_bytes = int(response.headers.get("content-length", 0))
    try:
        with (
            open(dest_path, "wb") as fh,
            tqdm(
                desc=dest_path.name,
                total=total_bytes,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                leave=False,
            ) as pbar,
        ):
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    fh.write(chunk)
                    pbar.update(len(chunk))
    except OSError as exc:
        logger.error("I/O error writing %s: %s", dest_path, exc)
        # Remove partially written file
        dest_path.unlink(missing_ok=True)
        return False

    logger.debug("Downloaded %s → %s", url, dest_path)
    return True


# ---------------------------------------------------------------------------
# 2. fetch_ncsu_metadata
# ---------------------------------------------------------------------------


def fetch_ncsu_metadata(config: Dict, force: bool = False) -> pd.DataFrame:
    """Download *metadata.csv* from the NCSU GitHub repository.

    Saves the file to ``data/metadata/ncsu_metadata.csv`` relative to the
    project root defined in *config['project_root']*.  If the cached file
    already exists and *force* is ``False``, the cached version is returned
    without re-downloading.

    Parameters
    ----------
    config:
        Project configuration dict.  Must contain key ``"project_root"``
        (``str`` or ``Path``).
    force:
        When ``True``, always re-download even if the cache exists.

    Returns
    -------
    pd.DataFrame
        Parsed metadata DataFrame with columns: ``file_name``, ``language``,
        ``transcript``, ``case_details``, ``case_pdf``.
    """
    project_root = Path(config["project_root"])
    dest = project_root / "data" / "metadata" / "ncsu_metadata.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        logger.info("Using cached NCSU metadata: %s", dest)
        return pd.read_csv(dest)

    url = f"{_GITHUB_RAW_BASE}/{_METADATA_REMOTE_PATH}"
    logger.info("Downloading NCSU metadata from %s", url)
    success = download_file(url, dest)
    if not success:
        raise RuntimeError(f"Failed to download NCSU metadata from {url}")

    df = pd.read_csv(dest)
    logger.info("Fetched %d metadata records.", len(df))
    return df


# ---------------------------------------------------------------------------
# 3. get_audio_file_list_from_github
# ---------------------------------------------------------------------------


def get_audio_file_list_from_github(config: Dict) -> List[str]:
    """Retrieve the list of ``.wav`` file paths via the GitHub Tree API.

    Uses the recursive tree endpoint so that nested directories are also
    traversed.

    Parameters
    ----------
    config:
        Project configuration dict (unused currently; included for
        consistency and future API-token support).

    Returns
    -------
    List[str]
        File paths (relative to the repository root) for every ``.wav``
        file found in the repository.
    """
    headers: Dict[str, str] = {"Accept": "application/vnd.github+json"}
    # Optional: support a personal access token to raise rate limits
    token: Optional[str] = config.get("github_token")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        response = requests.get(_GITHUB_TREE_API, headers=headers, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        logger.error("Failed to query GitHub Tree API: %s", exc)
        return []

    tree = response.json().get("tree", [])
    wav_paths = [
        item["path"]
        for item in tree
        if item.get("type") == "blob" and item["path"].endswith(".wav")
    ]
    logger.info("Found %d .wav files in the repository.", len(wav_paths))
    return wav_paths


# ---------------------------------------------------------------------------
# 4. download_audio_files
# ---------------------------------------------------------------------------


def download_audio_files(
    config: Dict,
    metadata_df: pd.DataFrame,
    max_files: Optional[int] = None,
) -> Dict[str, str]:
    """Download the audio files referenced in *metadata_df*.

    Only the ``_left.wav`` (robocaller channel) files are relevant; the
    function looks for both the base name listed in ``file_name`` and the
    corresponding ``_left.wav`` variant.

    Audio files are saved under ``data/raw/`` relative to the project root.

    Parameters
    ----------
    config:
        Project configuration dict (must contain ``"project_root"``).
    metadata_df:
        DataFrame returned by :func:`fetch_ncsu_metadata`.
    max_files:
        If set, limits the total number of files downloaded (useful for
        quick smoke-tests).

    Returns
    -------
    Dict[str, str]
        Mapping of ``file_name`` (as it appears in *metadata_df*) to the
        absolute local path of the downloaded file.  Failed downloads are
        omitted from the mapping.
    """
    project_root = Path(config["project_root"])
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Retrieve the full list of .wav paths available in the repository
    available_wavs: List[str] = get_audio_file_list_from_github(config)
    available_set = set(available_wavs)

    audio_paths: Dict[str, str] = {}
    count = 0

    file_names: List[str] = metadata_df["file_name"].dropna().tolist()

    for file_name in file_names:
        if max_files is not None and count >= max_files:
            logger.info("Reached max_files limit (%d). Stopping.", max_files)
            break

        # Prefer _left.wav; fall back to the raw file_name
        stem = Path(file_name).stem
        left_rel = f"{_AUDIO_GITHUB_DIR}/{stem}_left.wav"
        base_rel = f"{_AUDIO_GITHUB_DIR}/{file_name}"

        if left_rel in available_set:
            repo_rel = left_rel
        elif base_rel in available_set:
            repo_rel = base_rel
        else:
            logger.warning(
                "Audio not found in repo for file_name=%s. Skipping.", file_name
            )
            continue

        dest = raw_dir / Path(repo_rel).name
        if dest.exists():
            logger.debug("Already downloaded: %s", dest)
            audio_paths[file_name] = str(dest.resolve())
            count += 1
            continue

        url = f"{_GITHUB_RAW_BASE}/{repo_rel}"
        logger.info("[%d] Downloading %s", count + 1, url)
        success = download_file(url, dest)
        if success:
            audio_paths[file_name] = str(dest.resolve())
            count += 1
        else:
            logger.warning("Failed to download audio for file_name=%s.", file_name)

    logger.info(
        "Downloaded %d / %d audio files.", len(audio_paths), len(file_names)
    )
    return audio_paths


# ---------------------------------------------------------------------------
# 5. build_manifest
# ---------------------------------------------------------------------------


def build_manifest(
    config: Dict,
    metadata_df: pd.DataFrame,
    audio_paths: Dict[str, str],
) -> pd.DataFrame:
    """Build a standardised manifest DataFrame for the NCSU dataset.

    Parameters
    ----------
    config:
        Project configuration dict.
    metadata_df:
        Raw metadata DataFrame.
    audio_paths:
        Mapping of ``file_name`` → local audio path from
        :func:`download_audio_files`.

    Returns
    -------
    pd.DataFrame
        Manifest with columns: ``call_id``, ``audio_path``,
        ``audio_available``, ``transcript``, ``language``, ``label``,
        ``label_status``, ``source``, ``recording_type``, ``case_pdf``.
    """
    records = []
    for _, row in metadata_df.iterrows():
        file_name: str = str(row.get("file_name", ""))
        call_id: str = Path(file_name).stem if file_name else ""
        local_path: Optional[str] = audio_paths.get(file_name)

        records.append(
            {
                "call_id": call_id,
                "audio_path": local_path,
                "audio_available": local_path is not None,
                "transcript": row.get("transcript"),
                "language": row.get("language"),
                "label": "fraud",
                "label_status": "verified",
                "source": "ncsu_ftc",
                "recording_type": "robocall",
                "case_pdf": row.get("case_pdf"),
            }
        )

    manifest_df = pd.DataFrame(records)
    logger.info(
        "Manifest built: %d total records, %d with local audio.",
        len(manifest_df),
        manifest_df["audio_available"].sum(),
    )
    return manifest_df


# ---------------------------------------------------------------------------
# 6. save_manifest
# ---------------------------------------------------------------------------


def save_manifest(manifest_df: pd.DataFrame, config: Dict) -> Path:
    """Persist *manifest_df* to ``data/metadata/manifest.csv``.

    Parameters
    ----------
    manifest_df:
        Manifest DataFrame produced by :func:`build_manifest`.
    config:
        Project configuration dict (must contain ``"project_root"``).

    Returns
    -------
    Path
        Absolute path to the saved CSV file.
    """
    project_root = Path(config["project_root"])
    out_path = project_root / "data" / "metadata" / "manifest.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(out_path, index=False)
    logger.info("Manifest saved to %s (%d rows).", out_path, len(manifest_df))
    return out_path.resolve()


# ---------------------------------------------------------------------------
# 7. load_manifest
# ---------------------------------------------------------------------------


def load_manifest(config: Dict) -> pd.DataFrame:
    """Load an existing manifest CSV from disk.

    Parameters
    ----------
    config:
        Project configuration dict (must contain ``"project_root"``).

    Returns
    -------
    pd.DataFrame
        Manifest DataFrame.

    Raises
    ------
    FileNotFoundError
        If the manifest CSV does not exist.
    """
    project_root = Path(config["project_root"])
    manifest_path = project_root / "data" / "metadata" / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. Run ingestion first."
        )
    df = pd.read_csv(manifest_path)
    logger.info("Loaded manifest from %s (%d rows).", manifest_path, len(df))
    return df


# ---------------------------------------------------------------------------
# 8. run_ingestion
# ---------------------------------------------------------------------------


def run_ingestion(
    config: Dict,
    max_files: Optional[int] = None,
    force: bool = False,
) -> pd.DataFrame:
    """Orchestrate the full NCSU dataset ingestion pipeline.

    Steps:
    1. Fetch (or load cached) metadata CSV.
    2. Download audio files referenced in the metadata.
    3. Build a standardised manifest.
    4. Save the manifest to disk.

    Parameters
    ----------
    config:
        Project configuration dict (must contain ``"project_root"``).
    max_files:
        Optional cap on the number of audio files to download.
    force:
        When ``True``, re-download even if cached copies exist.

    Returns
    -------
    pd.DataFrame
        The resulting manifest DataFrame.
    """
    logger.info("=== NCSU Dataset Ingestion Start ===")

    # Step 1 – Metadata
    metadata_df: pd.DataFrame = fetch_ncsu_metadata(config, force=force)
    logger.info("Metadata loaded: %d records.", len(metadata_df))

    # Step 2 – Audio
    audio_paths: Dict[str, str] = download_audio_files(
        config, metadata_df, max_files=max_files
    )

    # Step 3 – Manifest
    manifest_df: pd.DataFrame = build_manifest(config, metadata_df, audio_paths)

    # Step 4 – Persist
    save_manifest(manifest_df, config)

    logger.info("=== NCSU Dataset Ingestion Complete ===")
    return manifest_df
