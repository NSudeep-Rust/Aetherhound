"""Audio DSP: bandpass + mains notches, then STFT spectrogram extraction."""
from __future__ import annotations

import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal

from config_loader import load_config

_FILTER_CACHE: dict[tuple, np.ndarray] = {}


def build_filter(sample_rate: float, lo_hz: float, hi_hz: float,
                 notch_base_hz: float, notch_harmonics: int, notch_q: float):
    key = (sample_rate, lo_hz, hi_hz, notch_base_hz, notch_harmonics, notch_q)
    if key in _FILTER_CACHE:
        return _FILTER_CACHE[key]
    nyquist = sample_rate / 2
    hi = min(hi_hz, nyquist - 50)
    sos = signal.butter(6, [lo_hz, hi], btype="bandpass", fs=sample_rate, output="sos")
    for k in range(1, notch_harmonics + 1):
        f = notch_base_hz * k
        if f < nyquist - 20:
            b, a = signal.iirnotch(f, Q=notch_q, fs=sample_rate)
            sos = np.vstack([sos, signal.tf2sos(b, a)])
    _FILTER_CACHE[key] = sos
    return sos


def apply_filters(audio: np.ndarray, sample_rate: float, band, zero_phase: bool = True) -> np.ndarray:
    sos = build_filter(sample_rate, band.lo_hz, band.hi_hz,
                       band.notch_base_hz, band.notch_harmonics, band.notch_q)
    audio = np.asarray(audio, dtype=np.float64)
    if audio.size <= 3 * max(len(sos), 1):
        return signal.sosfilt(sos, audio)
    return signal.sosfiltfilt(sos, audio) if zero_phase else signal.sosfilt(sos, audio)


def spectrogram_from_array(audio: np.ndarray, sample_rate: float,
                           nfft: int = 512, hop: int = 32) -> np.ndarray:
    _, _, spec = signal.stft(audio, fs=sample_rate, nperseg=nfft, noverlap=nfft - hop)
    return np.abs(spec).T


class StreamFilter:
    """Stateful version of apply_filters for chunked live input.

    Carries sosfilt filter state across chunks so the impulse response
    does not restart at every chunk boundary (which caused broadband
    transients that the autoencoder read as anomalies).
    """
    def __init__(self, sample_rate: float, band):
        self.sos = build_filter(sample_rate, band.lo_hz, band.hi_hz,
                                band.notch_base_hz, band.notch_harmonics, band.notch_q)
        self.zi = None

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float64)
        if self.zi is None:
            self.zi = signal.sosfilt_zi(self.sos) * x[0]
        y, self.zi = signal.sosfilt(self.sos, x, zi=self.zi)
        return y


def extract_spectrogram(file_path, nfft: int = 512, hop: int = 32,
                        band=None, apply_filters_flag: bool = True) -> np.ndarray:
    sample_rate, audio_data = wav.read(file_path)
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]

    if np.issubdtype(audio_data.dtype, np.integer):
        audio = audio_data.astype(np.float32) / float(np.iinfo(audio_data.dtype).max)
    else:
        audio = audio_data.astype(np.float32)

    if apply_filters_flag:
        band = band or load_config().band
        audio = apply_filters(audio, sample_rate, band)

    return spectrogram_from_array(audio, sample_rate, nfft, hop)


if __name__ == "__main__":
    import os
    print("[*] Testing DSP Pipeline extraction on mock files...")
    
    # Process the normal room baseline file we generated
    baseline_matrix = extract_spectrogram("data/baseline/normal_room.wav")
    print(f"[+] Baseline Spectrogram Matrix Shape: {baseline_matrix.shape}")
    
    # Verify shape dimensions are correct
    assert len(baseline_matrix.shape) == 2, "DSP output must be a 2D matrix!"
    print("[+] DSP Pipeline verified successfully!")
