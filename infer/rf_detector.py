"""RF anomaly detection against a calibrated per-bin baseline."""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import numpy as np

from dsp.rf_pipeline import bin_hz, peak_in_band


class RFAnomalyDetector:
    def __init__(self, cfg, db_path: str):
        self.cfg = cfg
        self.db_path = db_path
        path = Path(cfg.baseline_path)
        if not path.exists():
            raise FileNotFoundError(
                f"no RF baseline at {path}; run: python capture/record_rf.py"
            )
        data = np.load(path)
        self.mean_db = data["mean_db"]
        self.std_db = data["std_db"]
        self.bin_width = bin_hz(cfg.sample_rate_hz, cfg.nfft)
        self.start_hz = cfg.center_hz - cfg.sample_rate_hz / 2

        self.state = "CLEAN"
        self.hot = 0
        self.cold = 0
        self.last_peak_hz = 0.0
        self.last_z = 0.0
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
                "INSERT INTO rf_events (timestamp, peak_z, peak_freq_hz, current_state) VALUES (?,?,?,?)",
                (time.time(), float(z), float(freq), self.state),
            )

    def score(self, spectrum_db: np.ndarray) -> tuple[float, float, str]:
        if spectrum_db.size != self.mean_db.size:
            raise ValueError(
                f"bin count mismatch: {spectrum_db.size} vs baseline {self.mean_db.size}"
            )

        z = (spectrum_db - self.mean_db) / self.std_db
        peak_bin, peak_hz = peak_in_band(
            z, self.start_hz, self.bin_width,
            self.cfg.peak_search_lo_hz, self.cfg.peak_search_hi_hz,
        )
        peak_z = float(z[peak_bin])
        bins_over = int(np.count_nonzero(z > self.cfg.z_threshold))
        anomalous = (
            peak_z >= self.cfg.z_threshold and bins_over >= self.cfg.min_bins_over
        )

        self.last_z = peak_z
        self.last_peak_hz = peak_hz

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