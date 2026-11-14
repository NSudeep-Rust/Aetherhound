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
        self.callback_status_count = 0
        self.queue_drop_count = 0

    def _callback(self, indata, frames, time_info, status):
        if status:
            self.callback_status_count += 1
        block = indata[:, 0].copy()
        try:
            self._q.put_nowait(block)
        except queue.Full:
            try:
                self._q.get_nowait()
                self._q.put_nowait(block)
                self.queue_drop_count += 1
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

    def drain(self) -> list[np.ndarray]:
        """Return every queued microphone block without adding a stream gap.

        A 48 kHz / 1024-sample callback produces about 47 blocks per second,
        which is faster than the 30 Hz dashboard.  The runner must consume all
        available blocks on each frame instead of falling behind one block at a
        time.
        """
        blocks: list[np.ndarray] = []
        while True:
            try:
                blocks.append(self._q.get_nowait())
            except queue.Empty:
                return blocks
