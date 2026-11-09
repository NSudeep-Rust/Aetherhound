"""RSSI-peak direction finding with a switched directional antenna.

Not TDOA. RTL-SDRs on separate clocks drift too much to correlate. This sweeps a
directional antenna through fixed positions and takes the RSSI maximum.
Multipath will bias it; treat the result as a sector, not a bearing.
"""
from __future__ import annotations

import math
import threading
import time

import numpy as np

from dsp.rf_pipeline import averaged_spectrum, bin_hz

try:
    from gpiozero import OutputDevice

    _HAVE_GPIO = True
except Exception:
    _HAVE_GPIO = False


def parabolic_offset(y0: float, y1: float, y2: float) -> float:
    denom = y0 - 2 * y1 + y2
    if abs(denom) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5))


class BearingEstimator:
    def __init__(self, src, cfg, rf_cfg, src_lock=None, pins: list | None = None):
        self.src = src
        self.cfg = cfg
        self.rf = rf_cfg
        self.lock = src_lock or threading.Lock()
        self.pins = []
        if _HAVE_GPIO and pins:
            self.pins = [OutputDevice(p) for p in pins]
        self.last_deg = 0.0
        self.last_conf = 0.0

    def _select(self, index: int) -> None:
        if self.pins:
            for i, pin in enumerate(self.pins):
                pin.value = i == index
        else:
            print(f"[?] switch antenna {index} ({self.cfg.positions[index]} deg) then Enter", end=" ")
            input()

    def _rssi_at(self, target_hz: float) -> float:
        bw = self.cfg.target_bandwidth_hz
        samples = int(self.rf.sample_rate_hz * 0.1)
        frames = [self.src.read_iq(samples) for _ in range(2)]
        spec = averaged_spectrum(frames, self.rf.nfft, self.rf.sample_rate_hz)
        start = self.src.center_hz - self.rf.sample_rate_hz / 2
        w = bin_hz(self.rf.sample_rate_hz, self.rf.nfft)
        lo = max(0, int((target_hz - bw / 2 - start) / w))
        hi = min(spec.size, int((target_hz + bw / 2 - start) / w) + 1)
        if hi <= lo:
            return float(spec.max())
        return float(spec[lo:hi].mean())

    def measure(self, target_hz: float) -> tuple[float, float]:
        n = len(self.cfg.positions)
        if n < 3:
            raise ValueError("bearing needs at least 3 antenna positions")

        with self.lock:
            self.src.set_center(target_hz)
            time.sleep(0.05)
            rssi = []
            for i in range(n):
                self._select(i)
                time.sleep(self.cfg.hold_s)
                rssi.append(self._rssi_at(target_hz))

        arr = np.array(rssi)
        best = int(np.argmax(arr))
        step = 360.0 / n
        delta = parabolic_offset(arr[best - 1], arr[best], arr[(best + 1) % n]) * step
        deg = (self.cfg.positions[best] + delta) % 360.0

        spread = float(arr.max() - arr.min())
        self.last_deg, self.last_conf = deg, spread
        return deg, spread


def polar_plot(deg: float, conf: float, positions: list, radius: int = 9) -> list[str]:
    """Coarse 8-point compass rose with the estimate marked."""
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    rows = []
    for r in range(radius, 0, -1):
        line = ""
        for c in range(-radius, radius + 1):
            d = math.hypot(c, r)
            if abs(d - radius) < 0.7:
                line += "·"
            elif r == 1 and c == 0:
                line += "+"
            elif deg is not None and abs(d - radius) < 1.6 and math.degrees(
                math.atan2(c, r)
            ) % 360 < 30:
                line += "◆"
            else:
                line += " "
        rows.append(line)
    rows.append("  ".join(dirs))
    rows.append(f"{deg:.0f}°  spread {conf:.1f} dB")
    return rows