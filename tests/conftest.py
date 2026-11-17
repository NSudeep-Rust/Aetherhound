"""Build the data/ fixtures the test suite needs, if they aren't there yet.

data/ is gitignored (it's all synthetic/regenerable), so a fresh clone has
none of it and most of tests/test_core.py, tests/test_virtual_detection.py
fail before they even get to an assertion. Generate it once, lazily, the
first time a test needs it -- same recipe as tools/bootstrap.sh, just
invoked in-process instead of as a shell step so `pytest -q` alone is enough
on a fresh checkout.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest

REQUIRED_DATA_FILES = [
    ROOT / "data" / "baseline" / "normal_room.wav",
    ROOT / "data" / "test" / "infected_attack.wav",
    ROOT / "data" / "baseline_patches.npy",
    ROOT / "data" / "test_patches.npy",
]


@pytest.fixture(scope="session", autouse=True)
def _ensure_data_fixtures():
    baseline_wav = ROOT / "data" / "baseline" / "normal_room.wav"
    attack_wav = ROOT / "data" / "test" / "infected_attack.wav"
    baseline_patches = ROOT / "data" / "baseline_patches.npy"
    attack_patches = ROOT / "data" / "test_patches.npy"

    if all(path.exists() for path in REQUIRED_DATA_FILES):
        return

    import os

    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        import mock.generate_signals as generate_signals
        import dsp.patch_generator as patch_generator

        if not baseline_wav.exists() or not attack_wav.exists():
            print("[conftest] generating data/baseline and data/test audio fixtures...")
            baseline_wav.parent.mkdir(parents=True, exist_ok=True)
            attack_wav.parent.mkdir(parents=True, exist_ok=True)
            import scipy.io.wavfile as wav

            sr, baseline_audio = generate_signals.generate_synthetic_data(duration_sec=20, infected=False, gain_range=(1.0, 1.0))
            wav.write(str(baseline_wav), sr, baseline_audio)
            sr, infected_audio = generate_signals.generate_synthetic_data(duration_sec=20, infected=True, gain_range=(1.0, 1.0))
            wav.write(str(attack_wav), sr, infected_audio)

        if not baseline_patches.exists() or not attack_patches.exists():
            print("[conftest] slicing audio into spectrogram patches...")
            patch_generator.generate_and_save_patches(str(baseline_wav), str(baseline_patches))
            patch_generator.generate_and_save_patches(str(attack_wav), str(attack_patches))
    finally:
        os.chdir(cwd)
