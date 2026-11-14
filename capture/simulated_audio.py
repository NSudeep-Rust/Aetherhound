"""Deterministic, hardware-free audio source for the AetherHound demo."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile


class SimulatedAudioSource:
    """Read one or more WAV fixtures as fixed-size virtual microphone blocks."""

    name = "simulated-audio"

    def __init__(self, cfg, paths: list[str | Path], *, loop: bool = True):
        if not paths:
            raise ValueError("at least one WAV path is required")
        self.cfg = cfg
        self.blocksize = int(cfg.blocksize)
        self.loop = loop
        self._audio = self._load(paths, int(cfg.rate))
        self._offset = 0
        self._started = False

    @staticmethod
    def _load(paths: list[str | Path], expected_rate: int) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for item in paths:
            path = Path(item)
            if not path.exists():
                raise FileNotFoundError(f"simulation fixture not found: {path}")
            rate, samples = wavfile.read(path)
            if rate != expected_rate:
                raise ValueError(
                    f"{path} is {rate} Hz, but the configured virtual microphone is "
                    f"{expected_rate} Hz"
                )
            if samples.ndim > 1:
                samples = samples[:, 0]
            if np.issubdtype(samples.dtype, np.integer):
                samples = samples.astype(np.float32) / np.iinfo(samples.dtype).max
            else:
                samples = samples.astype(np.float32)
            chunks.append(samples)
        combined = np.concatenate(chunks)
        if combined.size == 0:
            raise ValueError("simulation fixture is empty")
        return combined

    def start(self) -> None:
        self._offset = 0
        self._started = True

    def stop(self) -> None:
        self._started = False

    def read(self, timeout: float = 0.0) -> np.ndarray | None:
        """Return the next virtual microphone block; ``timeout`` is API parity."""
        del timeout
        if not self._started:
            raise RuntimeError("source not started")
        if self._offset >= self._audio.size and not self.loop:
            return None

        end = self._offset + self.blocksize
        if end <= self._audio.size:
            block = self._audio[self._offset : end]
            self._offset = end
            return block.copy()

        # Wrap exactly, without a discontinuity caused by short final blocks.
        head = self._audio[self._offset :]
        if not self.loop:
            self._offset = self._audio.size
            return np.pad(head, (0, self.blocksize - head.size))
        remainder = self.blocksize - head.size
        repeats = int(np.ceil(remainder / self._audio.size))
        tail = np.tile(self._audio, repeats)[:remainder]
        self._offset = remainder % self._audio.size
        return np.concatenate((head, tail)).astype(np.float32, copy=False)
