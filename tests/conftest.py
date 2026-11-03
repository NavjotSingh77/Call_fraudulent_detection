"""
Shared pytest fixtures for test suite.
"""

import sys
from pathlib import Path
import pytest

# Ensure Call_fraudulent_detection is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tests.fixtures.sample_data import (
    create_corrupt_audio_file,
    create_minimal_config,
    create_sample_manifest,
    create_synthetic_audio,
    save_synthetic_audio,
)


@pytest.fixture
def sample_manifest():
    return create_sample_manifest()


@pytest.fixture
def minimal_config():
    return create_minimal_config()


@pytest.fixture
def synthetic_audio():
    return create_synthetic_audio(duration_seconds=5.0)


@pytest.fixture
def tmp_audio_file(tmp_path):
    return save_synthetic_audio(tmp_path / "test_audio.wav", duration=5.0)


@pytest.fixture
def corrupt_audio_file(tmp_path):
    return create_corrupt_audio_file(tmp_path / "corrupt_audio.wav")
