"""RF spectrum maths: IQ -> averaged dB power spectrum, bin decimation, peak search."""
from __future__ import annotations

import numpy as np
import scipy.signal as signal


def power_spectrum_db(iq: np.ndarray, nfft: int, sample_rate: float, window: str = "hann") -> np.ndarray:
    """Welch-style averaged periodogram in dB, DC removed."""
    iq = np.asarray(iq, dtype=np.complex128)
    if iq.size < nfft:
        raise ValueError(f"need >= {nfft} samples, got {iq.size}")

    iq = iq - iq.mean()
    w = signal.get_window(window, nfft)
    segments = iq.size // nfft
    acc = np.zeros(nfft, dtype=np.float64)

    for i in range(segments):
        block = iq[i * nfft:(i + 1) * nfft] * w
        # Use fft for complex input, not rfft
        acc += np.abs(np.fft.fft(block)) ** 2

    acc /= segments
    # For complex FFT, we get nfft bins (not nfft//2+1)
    # Shift so DC is in center, then take positive frequencies
    acc = np.fft.fftshift(acc)
    # Full complex spectrum after fftshift: covers [center - rate/2, center + rate/2)
    acc = np.maximum(acc, 1e-20)
    return 10.0 * np.log10(acc)


def averaged_spectrum(frames, nfft: int, sample_rate: float) -> np.ndarray:
    if not frames:
        raise ValueError("no frames supplied")
    stack = np.vstack([power_spectrum_db(f, nfft, sample_rate) for f in frames])
    return stack.mean(axis=0)


def bin_hz(sample_rate: float, nfft: int) -> float:
    return sample_rate / nfft


def bin_to_freq(index: int, start_hz: float, bin_width: float) -> float:
    return start_hz + index * bin_width


def decimate_max(spec: np.ndarray, width: int) -> np.ndarray:
    """Block-max decimation to `width` columns, preserving narrow peaks."""
    if spec.size == width:
        return spec.astype(np.float64)
    if spec.size < width:
        return np.interp(
            np.linspace(0, spec.size - 1, width), np.arange(spec.size), spec
        )
    edges = np.linspace(0, spec.size, width + 1).astype(int)
    return np.array([spec[a:b].max() for a, b in zip(edges[:-1], edges[1:])])


def peak_in_band(spec: np.ndarray, start_hz: float, bin_width: float,
                 lo_hz: float, hi_hz: float) -> tuple[int, float]:
    """Return (bin index, frequency) of the strongest bin inside the search band."""
    lo = max(0, int((lo_hz - start_hz) / bin_width))
    hi = min(spec.size, int((hi_hz - start_hz) / bin_width) + 1)
    if hi <= lo:
        lo, hi = 0, spec.size
    idx = lo + int(np.argmax(spec[lo:hi]))
    return idx, bin_to_freq(idx, start_hz, bin_width)