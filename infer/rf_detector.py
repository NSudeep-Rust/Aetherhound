"""RF anomaly detection against a per-bin swept baseline."""
from __future__ import annotations

import sqlite3
import time
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
        self.std_db = baseline["std_db"]
        self.valid = baseline["valid"].astype(bool)
        self.freqs = baseline["freqs"]

        self.state = "CLEAN"
        self.hot = 0
        self.cold = 0
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rf_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    peak_z REAL,
                    peak_freq_hz REAL,
                    current_state TEXT
                )
            """)

    def _log(self, z: float, freq: float) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO rf_events (timestamp, peak_z, peak_freq_hz, current_state)"
                " VALUES (?,?,?,?)",
                (time.time(), float(z), float(freq), self.state),
            )

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
        if anomalous:
            self.cold = 0
            self.hot += 1
            if self.hot >= self.cfg.rf_armed_frames:
                self.state = "THREAT_DETECTED"
        else:
            self.hot = 0
            self.cold += 1
            if self.cold >= self.cfg.rf_clear_frames:
                self.state = "CLEAN"

        if before != self.state:
            self._log(peak_z, peak_hz)
        return peak_z, peak_hz, self.state