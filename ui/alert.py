"""Non-blocking alarm tone on a background thread."""
from __future__ import annotations

import threading
import time

import numpy as np
import sounddevice as sd


class Buzzer:
    def __init__(self, freq: float = 2200.0, rate: int = 44100,
                 cooldown_s: float = 2.0, device: int | None = None):
        self._alarm = False
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._cooldown = cooldown_s
        self._device = None if device in (None, -1) else device
        self._rate = rate

        t = np.linspace(0, 0.1, int(rate * 0.1), endpoint=False)
        tone = 0.25 * np.sin(2 * np.pi * freq * t).astype(np.float32)
        fade = int(rate * 0.005)
        tone[:fade] *= np.linspace(0, 1, fade)
        tone[-fade:] *= np.linspace(1, 0, fade)
        self._tone = tone.reshape(-1, 1)

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def set(self, active: bool) -> None:
        with self._lock:
            self._alarm = active

    def _run(self) -> None:
        try:
            stream = sd.OutputStream(samplerate=self._rate, channels=1,
                                     dtype="float32", device=self._device)
            stream.start()
        except Exception:
            return

        while not self._stop.is_set():
            with self._lock:
                active = self._alarm
            if active:
                stream.write(self._tone)
            else:
                time.sleep(0.05)

        stream.stop()
        stream.close()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)