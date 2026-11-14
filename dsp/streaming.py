"""Shared live-audio patch assembly for monitoring and room calibration."""
from __future__ import annotations

import numpy as np

from dsp.pipeline import StreamFilter, spectrogram_from_array


class StreamingPatchBuilder:
    """Turn fixed-size audio blocks into the exact patches scored at runtime."""

    def __init__(self, audio_cfg, band_cfg, dsp_cfg, detect_cfg):
        self._audio_cfg = audio_cfg
        self._dsp_cfg = dsp_cfg
        self._filter = StreamFilter(audio_cfg.rate, band_cfg)
        self._buffer = np.zeros(0, dtype=np.float64)
        self._patch_samples = dsp_cfg.nfft + (dsp_cfg.patch - 1) * dsp_cfg.hop
        self._warmup_blocks = max(
            0,
            int(np.ceil(detect_cfg.stream_warmup_s * audio_cfg.rate / audio_cfg.blocksize)),
        )
        self._blocks_seen = 0

    def push(self, block: np.ndarray) -> np.ndarray | None:
        """Consume one block and return a spectrogram patch when one is ready."""
        samples = np.ravel(np.asarray(block, dtype=np.float32))
        if samples.size == 0:
            return None

        filtered = self._filter(samples)
        self._blocks_seen += 1
        if self._blocks_seen <= self._warmup_blocks:
            return None
        self._buffer = np.concatenate((self._buffer, filtered))
        if self._buffer.size < self._patch_samples:
            return None

        chunk = self._buffer[-self._patch_samples :].copy()
        # Preserve the last FFT window so subsequent patches overlap exactly as
        # they do in the monitoring dashboard.
        self._buffer = self._buffer[-self._dsp_cfg.nfft :].copy()
        patch = spectrogram_from_array(
            chunk,
            self._audio_cfg.rate,
            self._dsp_cfg.nfft,
            self._dsp_cfg.hop,
        )
        if patch.shape[0] != self._dsp_cfg.patch:
            raise RuntimeError(
                f"streaming STFT produced {patch.shape[0]} frames; expected {self._dsp_cfg.patch}"
            )
        return patch
