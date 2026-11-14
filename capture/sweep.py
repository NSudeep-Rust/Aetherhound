"""Wideband sweep: retune across a band plan and stitch segments into one frame."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from dsp.rf_pipeline import averaged_spectrum, bin_hz


@dataclass
class SweepFrame:
    db: np.ndarray
    freqs: np.ndarray
    valid: np.ndarray
    band_names: list = field(default_factory=list)
    timestamp: float = 0.0

    @property
    def span_hz(self) -> float:
        return float(self.freqs[-1] - self.freqs[0])


def band_label(lo: float, hi: float) -> str:
    return f"{lo/1e6:.0f}-{hi/1e6:.0f} MHz"


def build_axis(band_plan, sample_rate: float, nfft: int) -> tuple[np.ndarray, np.ndarray]:
    width = bin_hz(sample_rate, nfft)
    lo = min(b[0] for b in band_plan)
    hi = max(b[1] for b in band_plan)
    n = int(round((hi - lo) / width)) + 1
    freqs = lo + np.arange(n) * width
    valid = np.zeros(n, dtype=bool)
    for plo, phi in band_plan:
        valid |= (freqs >= plo) & (freqs <= phi)
    return freqs, valid


def build_steps(
    band_plan,
    sample_rate: float,
    step_hz: float | None = None,
    edge_guard_fraction: float = 0.10,
) -> list[float]:
    """Return retune centres with contiguous coverage inside usable windows."""
    usable = sample_rate * (1 - 2 * edge_guard_fraction)
    if not 0 < usable <= sample_rate:
        raise ValueError("edge_guard_fraction leaves no usable RF window")
    stride = min(float(step_hz if step_hz is not None else usable), usable)
    if stride <= 0:
        raise ValueError("step_hz must be positive")
    centers: list[float] = []
    for lo, hi in band_plan:
        span = hi - lo
        if span <= usable:
            centers.append((lo + hi) / 2)
            continue
        c = lo + usable / 2
        final = hi - usable / 2
        while c < final:
            centers.append(c)
            c += stride
        # Always cover the top edge of the band
        if not centers or centers[-1] < final:
            centers.append(final)
    return centers


class SweepEngine:
    def __init__(self, src, sweep_cfg, rf_cfg, src_lock: threading.Lock | None = None):
        self.src = src
        self.cfg = sweep_cfg
        self.rf = rf_cfg
        self.lock = src_lock or threading.Lock()
        self.width = bin_hz(rf_cfg.sample_rate_hz, rf_cfg.nfft)
        self.freqs, self.valid = build_axis(
            sweep_cfg.band_plan, rf_cfg.sample_rate_hz, rf_cfg.nfft
        )
        self.centers = build_steps(
            sweep_cfg.band_plan,
            rf_cfg.sample_rate_hz,
            sweep_cfg.step_hz,
            sweep_cfg.edge_guard_fraction,
        )
        self.guard = max(0, sweep_cfg.dc_guard_bins)
        self.edge_bins = int(np.ceil(rf_cfg.nfft * sweep_cfg.edge_guard_fraction))

    def sweep_once(self) -> SweepFrame:
        acc = np.full(self.freqs.size, -200.0)
        seen = np.zeros(self.freqs.size, dtype=bool)
        lo = self.freqs[0]
        half = self.rf.sample_rate_hz / 2
        samples = max(int(self.rf.sample_rate_hz * self.cfg.dwell_ms / 1000.0), self.rf.nfft)

        with self.lock:
            for center in self.centers:
                self.src.set_center(center)
                time.sleep(0.002)  # PLL settle
                frames = [
                    self.src.read_iq(samples) for _ in range(self.rf.frames_per_spectrum)
                ]
                seg = averaged_spectrum(frames, self.rf.nfft, self.rf.sample_rate_hz)

                # Discard unreliable FFT window edges.  Retunes overlap by
                # construction, so valid interior regions still cover the band.
                edge = min(self.edge_bins, max(0, seg.size // 2 - 1))
                retained = seg[edge : seg.size - edge] if edge else seg
                start = center - half + edge * self.width
                i0 = int(round((start - lo) / self.width))
                i1 = i0 + retained.size
                a, b = max(i0, 0), min(i1, self.freqs.size)
                if b <= a:
                    continue
                sa, sb = a - i0, b - i0
                acc[a:b] = np.maximum(acc[a:b], retained[sa:sb])
                seen[a:b] = True

                dc = int(round((center - lo) / self.width))
                g0, g1 = max(dc - self.guard, 0), min(dc + self.guard + 1, self.freqs.size)
                if g1 > g0:
                    seen[g0:g1] = False

        valid = seen & self.valid
        names = [band_label(lo_, hi_) for lo_, hi_ in self.cfg.band_plan]
        return SweepFrame(acc, self.freqs, valid, names, time.time())


class SweepWorker(threading.Thread):
    """Runs sweeps continuously; the UI reads the latest snapshot."""

    def __init__(self, engine: SweepEngine, detector):
        super().__init__(daemon=True)
        self.engine = engine
        self.detector = detector
        self._halt = threading.Event()
        self._lock = threading.Lock()
        self.frame: SweepFrame | None = None
        self.z = 0.0
        self.peak_hz = 0.0
        self.state = "CLEAN"
        self.sweeps = 0

    def run(self) -> None:
        while not self._halt.is_set():
            try:
                frame = self.engine.sweep_once()
                z, peak_hz, state = self.detector.score(frame)
            except Exception as exc:
                self.state = f"ERR: {exc}"
                time.sleep(1.0)
                continue
            with self._lock:
                self.frame, self.z, self.peak_hz, self.state = frame, z, peak_hz, state
                self.sweeps += 1

    def snapshot(self):
        with self._lock:
            return self.frame, self.z, self.peak_hz, self.state, self.sweeps

    def stop(self) -> None:
        self._halt.set()
