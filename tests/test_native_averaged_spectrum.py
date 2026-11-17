"""Correctness tests for native/averaged_spectrum.c (via native/native_rf.py).

Skipped if the shared library hasn't been built yet (run ./native/build.sh),
same pattern as test_training_split.py skipping when torch isn't installed --
this is an opt-in benchmark artifact, not part of the default install.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

_LIB_PATH = Path(__file__).resolve().parent.parent / "native" / "libaveraged_spectrum.so"
if not _LIB_PATH.exists():
    pytest.skip(
        f"{_LIB_PATH} not built -- run ./native/build.sh to test the native averaged_spectrum",
        allow_module_level=True,
    )

from native.native_rf import averaged_spectrum_native  # noqa: E402


def _hann(n: int) -> np.ndarray:
    if n == 1:
        return np.array([1.0])
    k = np.arange(n)
    return 0.5 - 0.5 * np.cos(2 * np.pi * k / (n - 1))


def _reference_averaged_spectrum(frames, nfft: int) -> np.ndarray:
    """Same math as dsp.rf_pipeline.averaged_spectrum, without the scipy
    dependency (scipy.signal.get_window("hann", n) is this exact formula)."""
    def power_spectrum_db(iq):
        iq = np.asarray(iq, dtype=np.complex128)
        iq = iq - iq.mean()
        w = _hann(nfft)
        segments = iq.size // nfft
        blocks = iq[: segments * nfft].reshape(segments, nfft)
        acc = np.mean(np.abs(np.fft.fft(blocks * w, axis=1)) ** 2, axis=0)
        acc = np.fft.fftshift(acc)
        acc = np.maximum(acc, 1e-20)
        return 10.0 * np.log10(acc)

    return np.vstack([power_spectrum_db(f) for f in frames]).mean(axis=0)


@pytest.mark.parametrize(
    "nfft,n_frames,samples_per_frame",
    [(512, 1, 512 * 4), (2048, 5, 2048 * 3), (2048, 1, 2048), (256, 3, 256 * 2)],
)
def test_matches_reference_implementation(nfft, n_frames, samples_per_frame):
    rng = np.random.default_rng(0)
    frames = [
        rng.normal(0, 1, samples_per_frame) + 1j * rng.normal(0, 1, samples_per_frame)
        for _ in range(n_frames)
    ]
    expected = _reference_averaged_spectrum(frames, nfft)
    actual = averaged_spectrum_native(frames, nfft, sample_rate=2_400_000)
    assert actual.shape == (nfft,)
    assert np.allclose(expected, actual, atol=1e-8)


def test_rejects_empty_frames():
    with pytest.raises(ValueError):
        averaged_spectrum_native([], 512, 2_400_000)


def test_rejects_frame_shorter_than_nfft():
    with pytest.raises(ValueError):
        averaged_spectrum_native([np.zeros(100, dtype=complex)], 512, 2_400_000)


def test_rejects_non_power_of_two_nfft():
    with pytest.raises(ValueError):
        averaged_spectrum_native([np.zeros(600, dtype=complex)], 500, 2_400_000)


def test_rejects_mismatched_frame_lengths():
    with pytest.raises(ValueError):
        averaged_spectrum_native(
            [np.zeros(1024, dtype=complex), np.zeros(2048, dtype=complex)], 512, 2_400_000,
        )
