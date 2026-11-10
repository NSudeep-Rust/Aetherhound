"""RF front-end abstraction: RTL-SDR, HackRF, or a simulated source.

All backends return 1-D numpy arrays of complex64 IQ samples.
"""
from __future__ import annotations

import sys
import time

import numpy as np


class RFSource:
    name = "base"

    def open(self) -> None:
        raise NotImplementedError

    def set_center(self, hz: float) -> None:
        self.center_hz = hz

    def read_iq(self, n: int) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class RTLSDRBackend(RFSource):
    name = "rtlsdr"

    def __init__(self, index: int, center_hz: float, rate: float, gain_db: float, gain_auto: bool):
        self.index = index
        self.center_hz = center_hz
        self.rate = rate
        self.gain_db = gain_db
        self.gain_auto = gain_auto
        self._sdr = None

    def open(self) -> None:
        from rtlsdr import RtlSdr  # pyrtlsdr

        self._sdr = RtlSdr(device_index=self.index)
        self._sdr.sample_rate = self.rate
        self._sdr.center_freq = self.center_hz
        self._sdr.gain = "auto" if self.gain_auto else self.gain_db

    def set_center(self, hz: float) -> None:
        self.center_hz = hz
        if self._sdr is not None:
            self._sdr.center_freq = hz

    def read_iq(self, n: int) -> np.ndarray:
        if self._sdr is None:
            raise RuntimeError("source not open")
        return np.asarray(self._sdr.read_samples(n), dtype=np.complex64)

    def close(self) -> None:
        if self._sdr is not None:
            try:
                self._sdr.close()
            finally:
                self._sdr = None


class HackRFBackend(RFSource):
    name = "hackrf"

    def __init__(self, center_hz: float, rate: float, gain_db: float):
        self.center_hz = center_hz
        self.rate = rate
        self.gain_db = gain_db
        self._hackrf = None

    def open(self) -> None:
        from pyhackrf import HackRF

        self._hackrf = HackRF()
        self._hackrf.sample_rate = int(self.rate)
        self._hackrf.center_freq = int(self.center_hz)
        self._hackrf.gain = int(self.gain_db)
        self._hackrf.start_rx()

    def set_center(self, hz: float) -> None:
        self.center_hz = hz
        if self._hackrf is not None:
            self._hackrf.center_freq = int(hz)

    def read_iq(self, n: int) -> np.ndarray:
        if self._hackrf is None:
            raise RuntimeError("source not open")
        return np.asarray(self._hackrf.read_samples(n), dtype=np.complex64)

    def close(self) -> None:
        if self._hackrf is not None:
            try:
                self._hackrf.stop_rx()
                self._hackrf.close()
            finally:
                self._hackrf = None


class SimulatedBackend(RFSource):
    """Noise floor plus an intermittently injected narrowband carrier."""

    name = "simulated"

    def __init__(self, center_hz: float, rate: float, carrier_hz: float = 500e3):
        self.center_hz = center_hz
        self.rate = rate
        self.carrier_offset_hz = carrier_hz  # Store as offset from center
        self._n = 0
        self.inject_carrier = False

    def open(self) -> None:
        self._n = 0

    def set_center(self, hz: float) -> None:
        self.center_hz = hz

    def read_iq(self, n: int) -> np.ndarray:
        noise = (
            np.random.normal(0, 0.5, n) + 1j * np.random.normal(0, 0.5, n)
        ).astype(np.complex64)

        if self.inject_carrier:
            offset = self.carrier_offset_hz
            if abs(offset) < self.rate / 2:
                t = (np.arange(n) + self._n) / self.rate
                noise += (0.35 * np.exp(2j * np.pi * offset * t)).astype(np.complex64)

        self._n += n
        return noise


def open_rf_source(cfg) -> RFSource:
    """Resolve rf.backend, falling back to a simulated source."""
    choice = cfg.backend
    order = ["rtlsdr", "hackrf", "simulated"] if choice == "auto" else [choice]

    errors: list[str] = []
    for name in order:
        if name == "rtlsdr":
            src = RTLSDRBackend(
                cfg.device_index, cfg.center_hz, cfg.sample_rate_hz, cfg.gain_db, cfg.gain_auto
            )
        elif name == "hackrf":
            src = HackRFBackend(cfg.center_hz, cfg.sample_rate_hz, cfg.gain_db)
        elif name == "simulated":
            return SimulatedBackend(cfg.center_hz, cfg.sample_rate_hz)
        else:
            errors.append(f"unknown backend {name!r}")
            continue

        try:
            src.open()
            return src
        except Exception as exc:  # missing lib, no device, busy
            errors.append(f"{name}: {exc}")
            print(f"[!] RF backend {name} unavailable: {exc}", file=sys.stderr)

    raise RuntimeError("no RF source available: " + "; ".join(errors))