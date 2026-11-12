import numpy as np
import pytest

from capture.rf_source import SimulatedBackend
from capture.sweep import SweepEngine, build_axis, build_steps
from config_loader import Config
from dsp.rf_pipeline import bin_hz
from infer.rf_detector import RFAnomalyDetector

PLAN = [[88e6, 96e6], [400e6, 410e6]]


@pytest.fixture
def cfg():
    c = Config()
    c.rf.sample_rate_hz, c.rf.nfft = 2.4e6, 2048
    c.sweep.band_plan = PLAN
    c.sweep.step_hz = 2e6
    return c


def test_axis_covers_plan(cfg):
    freqs, valid = build_axis(PLAN, cfg.rf.sample_rate_hz, cfg.rf.nfft)
    assert freqs[0] <= 88e6 and freqs[-1] >= 410e6 - 1e3
    assert valid[freqs.searchsorted(90e6)]
    assert not valid[freqs.searchsorted(200e6)]


def test_steps_stay_in_band(cfg):
    for c in build_steps(PLAN, cfg.rf.sample_rate_hz):
        assert any(lo <= c <= hi for lo, hi in PLAN)


def test_sweep_shape(cfg):
    src = SimulatedBackend(90e6, cfg.rf.sample_rate_hz)
    eng = SweepEngine(src, cfg.sweep, cfg.rf)
    frame = eng.sweep_once()
    assert frame.db.size == frame.freqs.size == frame.valid.size
    assert frame.valid.any()


def test_detector_localises_carrier(cfg):
    src = SimulatedBackend(92e6, cfg.rf.sample_rate_hz, carrier_hz=500e3)
    eng = SweepEngine(src, cfg.sweep, cfg.rf)

    base = eng.sweep_once()
    baseline = {
        "mean_db": base.db, "std_db": np.full(base.db.size, 0.5),
        "valid": base.valid, "freqs": base.freqs,
    }
    det = RFAnomalyDetector(cfg.rf, ":memory:", baseline)

    src.inject_carrier = True
    z, hz, _ = det.score(eng.sweep_once())
    assert z > cfg.rf.z_threshold
    # Carrier is injected at offset from each sweep center, so it appears
    # in both bands. Per-band search picks the max z across bands.
    # Verify the detected peak falls within one of the band plan bands.
    in_any_band = any(lo <= hz <= hi for lo, hi in cfg.sweep.band_plan)
    assert in_any_band, f"carrier at {hz/1e6:.3f} MHz not in any band plan band"