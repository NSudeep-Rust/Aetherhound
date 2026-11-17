"""Honest before/after benchmark: native C averaged_spectrum vs. the
existing numpy/scipy implementation in dsp/rf_pipeline.py.

The project review profiled a live RF sweep and found FFT work is ~33% of
per-step time, with the simulator's own RNG (45%) and a fixed per-step sleep
(19%) accounting for the rest -- i.e. even a free, instant FFT could only
ever claw back about a third of one sweep step. This script measures
whether the native port is actually faster at this project's real sizes;
run it yourself with ``python native/bench_averaged_spectrum.py`` and read
off the printed numbers before deciding whether to wire native_rf.py into
anything. Build the library first: ./native/build.sh

nfft=512 matches the audio config (dsp.nfft); nfft=2048, frames_per_spectrum=4
matches the RF config (rf.nfft, rf.frames_per_spectrum) in config.toml.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from native.native_rf import averaged_spectrum_native

try:
    from dsp.rf_pipeline import averaged_spectrum as averaged_spectrum_numpy
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


def _hann(n: int) -> np.ndarray:
    if n == 1:
        return np.array([1.0])
    k = np.arange(n)
    return 0.5 - 0.5 * np.cos(2 * np.pi * k / (n - 1))


def _power_spectrum_db_numpy_only(iq: np.ndarray, nfft: int) -> np.ndarray:
    """Same math as dsp.rf_pipeline.power_spectrum_db, without the scipy
    dependency (scipy.signal.get_window("hann", n) == this Hann formula) --
    used as the correctness/speed reference when scipy isn't installed."""
    iq = np.asarray(iq, dtype=np.complex128)
    iq = iq - iq.mean()
    w = _hann(nfft)
    segments = iq.size // nfft
    blocks = iq[: segments * nfft].reshape(segments, nfft)
    acc = np.mean(np.abs(np.fft.fft(blocks * w, axis=1)) ** 2, axis=0)
    acc = np.fft.fftshift(acc)
    acc = np.maximum(acc, 1e-20)
    return 10.0 * np.log10(acc)


def _averaged_spectrum_numpy_only(frames, nfft: int) -> np.ndarray:
    stack = np.vstack([_power_spectrum_db_numpy_only(f, nfft) for f in frames])
    return stack.mean(axis=0)


def _time_it(fn, repeats: int) -> float:
    # One untimed warmup call so Python import/first-call overhead doesn't
    # skew the measurement.
    fn()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t0) / repeats


def bench(nfft: int, n_frames: int, samples_per_frame: int, repeats: int = 20) -> None:
    rng = np.random.default_rng(0)
    frames = [
        rng.normal(0, 1, samples_per_frame) + 1j * rng.normal(0, 1, samples_per_frame)
        for _ in range(n_frames)
    ]

    if _HAVE_SCIPY:
        numpy_fn = lambda: averaged_spectrum_numpy(frames, nfft, 2_400_000)
        label = "numpy+scipy (dsp.rf_pipeline, production path)"
    else:
        numpy_fn = lambda: _averaged_spectrum_numpy_only(frames, nfft)
        label = "numpy-only reference (scipy unavailable here; identical math, see docstring)"

    native_fn = lambda: averaged_spectrum_native(frames, nfft, 2_400_000)

    ref = numpy_fn()
    native = native_fn()
    max_abs_diff = float(np.max(np.abs(ref - native)))

    numpy_s = _time_it(numpy_fn, repeats)
    native_s = _time_it(native_fn, repeats)
    speedup = numpy_s / native_s if native_s > 0 else float("inf")

    print(f"nfft={nfft:5d}  n_frames={n_frames:2d}  samples/frame={samples_per_frame:7d}")
    print(f"    {label}: {numpy_s * 1e3:8.4f} ms/call")
    print(f"    native C (radix-2 FFT, ctypes):                       {native_s * 1e3:8.4f} ms/call")
    print(f"    native/numpy time ratio: {native_s / numpy_s:.2f}x "
          f"({'native faster' if speedup > 1 else 'numpy faster or comparable'})")
    print(f"    max|dB diff| vs numpy: {max_abs_diff:.3e} (correctness check)")
    print()


if __name__ == "__main__":
    if not _HAVE_SCIPY:
        print("[!] scipy not importable in this environment -- comparing against a numpy-only\n"
              "    reference with identical math instead of dsp.rf_pipeline's production path.\n"
              "    Re-run this on a machine with the project's real requirements installed for\n"
              "    the production-path numbers.\n")
    print("[*] nfft=512 matches config.toml's [dsp].nfft (audio)")
    bench(nfft=512, n_frames=1, samples_per_frame=512 * 8)
    print("[*] nfft=2048, n_frames=4 matches config.toml's [rf].nfft / frames_per_spectrum")
    bench(nfft=2048, n_frames=4, samples_per_frame=2048 * 8)
    print("[*] A single larger RF capture (more segments per frame)")
    bench(nfft=2048, n_frames=4, samples_per_frame=2048 * 64)
