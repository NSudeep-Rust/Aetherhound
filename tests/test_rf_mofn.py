"""RF M-of-N confirmation: a bursty/duty-cycled carrier must still escalate.

Before this fix, infer/rf_detector.py required rf_armed_frames consecutive
anomalous sweeps to go ARMED -> THREAT_DETECTED (a single miss reset the
counter to 0). A carrier that is not guaranteed to show up on every sweep
(duty-cycled transmission, sweep timing not aligned with the burst) could
then sit in ARMED forever. These tests exercise the real RFAnomalyDetector
state machine against synthetic sweeps -- no SDR, GNU Radio, or scipy-backed
DSP needed, just the frame fields RFAnomalyDetector.score() reads.
"""
from __future__ import annotations

import numpy as np

from capture.sweep import SweepFrame
from config_loader import Config
from infer.rf_detector import RFAnomalyDetector


def _frame(freqs: np.ndarray, db: np.ndarray) -> SweepFrame:
    return SweepFrame(db=db, freqs=freqs, valid=np.ones_like(db, dtype=bool))


def _make_detector(**rf_overrides) -> tuple[RFAnomalyDetector, np.ndarray, np.ndarray]:
    n = 50
    freqs = np.linspace(100e6, 110e6, n)
    mean_db = np.zeros(n)
    baseline = {
        "mean_db": mean_db,
        "std_db": np.ones(n),
        "valid": np.ones(n, dtype=bool),
        "freqs": freqs,
    }

    cfg = Config().rf
    cfg.band_plan = [[100e6, 110e6]]
    cfg.z_threshold = 6.0
    cfg.min_bins_over = 1
    cfg.rf_armed_frames = 3
    cfg.rf_window_frames = 10
    cfg.rf_clear_frames = 15
    for key, value in rf_overrides.items():
        setattr(cfg, key, value)

    det = RFAnomalyDetector(cfg, ":memory:", baseline)
    return det, freqs, mean_db


def test_bursty_carrier_escalates_with_m_of_n_confirmation():
    det, freqs, mean_db = _make_detector()
    carrier_bin = 25

    hot_db = mean_db.copy()
    hot_db[carrier_bin] = 10.0  # +10 sigma against std_db=1 -> z=10 > z_threshold=6
    cold_db = mean_db.copy()

    # Duty-cycled carrier: on every other sweep. A strict "3 in a row" rule
    # (the old self.hot counter, reset to 0 on any miss) would never see 3
    # consecutive hits in this pattern and would be stuck in ARMED forever.
    states = []
    for i in range(10):
        frame_db = hot_db if i % 2 == 0 else cold_db
        _, _, state = det.score(_frame(freqs, frame_db))
        states.append(state)

    assert "THREAT_DETECTED" in states, f"bursty carrier never escalated past ARMED: {states}"


def test_isolated_single_hit_does_not_escalate():
    det, freqs, mean_db = _make_detector()
    carrier_bin = 25

    hot_db = mean_db.copy()
    hot_db[carrier_bin] = 10.0
    cold_db = mean_db.copy()

    # One stray hit still arms (unchanged from before), but with only 1 hit
    # ever in the trailing window it must not satisfy rf_armed_frames=3.
    states = [det.score(_frame(freqs, hot_db))[2]]
    states += [det.score(_frame(freqs, cold_db))[2] for _ in range(9)]

    assert "THREAT_DETECTED" not in states, f"a single isolated hit escalated: {states}"
    assert states[-1] == "ARMED"
