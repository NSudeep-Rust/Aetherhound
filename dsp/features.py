"""Small, explainable signal features used beside the audio autoencoder."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter


def tonal_prominence(
    patch: np.ndarray,
    sample_rate: float,
    nfft: int,
    *,
    min_hz: float = 200.0,
    max_hz: float = 18_000.0,
    window_bins: int = 17,
) -> tuple[float, float]:
    """Return the strongest narrow-band line's prominence and frequency.

    ``patch`` is a magnitude spectrogram (time x frequency).  A covert acoustic
    channel is commonly a steady carrier switched on and off to encode bits.  It
    therefore appears as a tall, local spectral line, while ordinary room noise
    remains broad-band.  Measuring each bin against a local median makes this
    feature insensitive to microphone gain and to the broad shape of the room.
    """
    matrix = np.asarray(patch, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < 3:
        raise ValueError("patch must be a non-empty (time, frequency) spectrogram")
    if nfft <= 0 or sample_rate <= 0:
        raise ValueError("sample_rate and nfft must be positive")
    if window_bins < 3 or window_bins % 2 == 0:
        raise ValueError("window_bins must be an odd integer of at least 3")

    # Averaging over the short patch suppresses random noise but retains a
    # carrier.  Work in dB so the score is a directly readable prominence.
    spectrum_db = 20.0 * np.log10(np.maximum(np.mean(matrix, axis=0), 1e-12))
    local_floor_db = median_filter(spectrum_db, size=window_bins, mode="nearest")
    prominence_db = spectrum_db - local_floor_db

    bin_width_hz = sample_rate / nfft
    lo = max(1, int(np.ceil(min_hz / bin_width_hz)))
    hi = min(matrix.shape[1] - 1, int(np.floor(max_hz / bin_width_hz)))
    if lo > hi:
        raise ValueError("requested frequency range has no FFT bins")

    relative_index = int(np.argmax(prominence_db[lo : hi + 1]))
    index = lo + relative_index
    return float(prominence_db[index]), float(index * bin_width_hz)
