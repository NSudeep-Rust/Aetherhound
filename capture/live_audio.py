"""Live microphone capture into a bounded block queue."""
from __future__ import annotations

import queue

import numpy as np
import sounddevice as sd


class LiveAudioSource:
    def __init__(self, cfg):
        self.cfg = cfg
        self.blocksize = cfg.blocksize
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)
        self._stream = None

    def _callback(self, indata, frames, time_info, status):
        block = indata[:, 0].copy()
        try:
            self._q.put_nowait(block)
        except queue.Full:
            try:
                self._q.get_nowait()
                self._q.put_nowait(block)
            except queue.Empty:
                pass

    def start(self) -> None:
        device = None if self.cfg.device < 0 else self.cfg.device
        self._stream = sd.InputStream(
            samplerate=self.cfg.rate, channels=self.cfg.channels,
            blocksize=self.blocksize, dtype="float32",
            device=device, callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def read(self, timeout: float = 0.5) -> np.ndarray | None:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None