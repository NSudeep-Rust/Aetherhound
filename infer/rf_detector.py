"""RF anomaly detection against a per-bin swept baseline."""
from __future__ import annotations

import sqlite3
import time
from collections import deque
from pathlib import Path

import numpy as np


class RFAnomalyDetector:
    def __init__(self, cfg, db_path: str, baseline: dict | None = None):
        self.cfg = cfg
        self.db_path = db_path

        if baseline is None:
            path = Path(cfg.baseline_path)
            if not path.exists():
                raise FileNotFoundError(
                    f"no RF baseline at {path}; run: python capture/record_rf.py"
                )
            baseline = dict(np.load(path))

        self.mean_db = baseline["mean_db"]
        self.std_db = np.maximum(np.asarray(baseline["std_db"], dtype=np.float64), 0.5)
        self.valid = baseline["valid"].astype(bool)
        self.freqs = baseline["freqs"]

        self.state = "CLEAN"
        # M-of-N confirmation: ARMED -> THREAT_DETECTED once rf_armed_frames
        # hits land within the trailing rf_window_frames sweeps, rather than
        # requiring them consecutively. A duty-cycled/bursty carrier is not
        # guaranteed to show up on every sweep -- a strict "N in a row" rule
        # (the old self.hot counter, reset to 0 on any single miss) can leave
        # it stuck in ARMED forever even though it reliably reappears.
        window_frames = int(getattr(cfg, "rf_window_frames", cfg.rf_armed_frames))
        self.hit_window: deque[bool] = deque(maxlen=window_frames)
        self.cold = 0
        # A single long-lived connection, not one per call: sqlite3.connect(
        # ":memory:") opens a brand-new, unshared in-memory database on every
        # call, so a connect-per-call pattern silently loses the table created
        # in _init_db() the moment :memory: is used (as every test here does).
        # check_same_thread=False because SweepWorker scores frames, and so
        # logs, on a background thread while this connection is created here
        # on the caller's thread.
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self) -> None:
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS rf_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                peak_z REAL,
                peak_freq_hz REAL,
                current_state TEXT
            )
        """)
        self._conn.commit()

    def _log(self, z: float, freq: float) -> None:
        self._conn.execute(
            "INSERT INTO rf_events (timestamp, peak_z, peak_freq_hz, current_state)"
            " VALUES (?,?,?,?)",
            (time.time(), float(z), float(freq), self.state),
        )
        self._conn.commit()

    def check_baseline_freshness(self, frame, max_median_z: float = 4.0) -> str | None:
        """Sanity-check the loaded baseline against one live sweep at startup.

        A placeholder or stale baseline (e.g. a flat -50 dB stub, or one
        recorded before a hardware/antenna change) makes every live bin look
        like an outlier, which either floods THREAT_DETECTED on an idle room
        or silently normalizes out a real attack. Call this once before the
        detection loop starts; it does not affect scoring.
        """
        if frame.db.size != self.mean_db.size:
            return (
                f"RF baseline shape mismatch: frame has {frame.db.size} bins, "
                f"baseline has {self.mean_db.size}. Re-run capture/record_rf.py."
            )
        mask = frame.valid & self.valid
        if not mask.any():
            return "RF baseline shares no valid bins with a live sweep; re-run capture/record_rf.py."
        z = (frame.db[mask] - self.mean_db[mask]) / self.std_db[mask]
        median_z = float(np.median(np.abs(z)))
        if median_z > max_median_z:
            return (
                f"RF baseline looks stale: a live sweep's median |z-score| is "
                f"{median_z:.1f} (expected well under {max_median_z:.0f} on a quiet "
                "room). Re-run capture/record_rf.py and commit the result."
            )
        return None

    def score(self, frame) -> tuple[float, float, str]:
        if frame.db.size != self.mean_db.size:
            raise ValueError(f"frame width {frame.db.size} != baseline {self.mean_db.size}")

        mask = frame.valid & self.valid
        z = np.full(frame.db.size, -np.inf)
        z[mask] = (frame.db[mask] - self.mean_db[mask]) / self.std_db[mask]

        # Per-band peak search: find peak within each band, then take the max across bands
        # This prevents the "max over 2M bins" problem
        band_plan = self.cfg.band_plan if hasattr(self.cfg, 'band_plan') else []
        if not band_plan:
            # Fallback to old behavior if no band_plan
            band = (frame.freqs >= self.cfg.peak_search_lo_hz) & (
                frame.freqs <= self.cfg.peak_search_hi_hz
            )
            search = mask & band
            if not search.any():
                search = mask
        else:
            # Search within each band separately
            search = np.zeros_like(mask, dtype=bool)
            for lo, hi in band_plan:
                band_mask = (frame.freqs >= lo) & (frame.freqs <= hi) & mask
                if band_mask.any():
                    # Find peak in this band
                    band_z = np.where(band_mask, z, -np.inf)
                    band_idx = int(np.argmax(band_z))
                    search[band_idx] = True
        
        if not search.any():
            return 0.0, 0.0, self.state

        idx = int(np.argmax(np.where(search, z, -np.inf)))
        peak_z = float(z[idx])
        peak_hz = float(frame.freqs[idx])
        
        # Count bins_over within the band containing the peak (not across all bands)
        # Find which band the peak is in
        bins_over = 0
        for lo, hi in band_plan:
            if lo <= peak_hz <= hi:
                band_mask = (frame.freqs >= lo) & (frame.freqs <= hi) & mask
                bins_over = int(np.count_nonzero(z[band_mask] > self.cfg.z_threshold))
                break
        else:
            # Fallback if peak not in any band
            bins_over = int(np.count_nonzero(z[search] > self.cfg.z_threshold))
        
        anomalous = peak_z >= self.cfg.z_threshold and bins_over >= self.cfg.min_bins_over

        before = self.state
        self.hit_window.append(anomalous)
        if anomalous:
            self.cold = 0
            if self.state == "CLEAN":
                self.state = "ARMED"
            if self.state == "ARMED" and sum(self.hit_window) >= self.cfg.rf_armed_frames:
                self.state = "THREAT_DETECTED"
        else:
            self.cold += 1
            if self.cold >= self.cfg.rf_clear_frames:
                self.state = "CLEAN"
                self.hit_window.clear()

        if before != self.state:
            self._log(peak_z, peak_hz)
        return peak_z, peak_hz, self.state
