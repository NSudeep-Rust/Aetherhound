import numpy as np
import pytest

from capture.rf_source import SimulatedBackend
from config_loader import Config
from dsp.rf_pipeline import averaged_spectrum, decimate_max, peak_in_band, power_spectrum_db
from ui.waterfall import Waterfall


@pytest.fixture
def cfg():
    c = Config()
    c.rf.center_hz, c.rf.sample_rate_hz = 100e6, 2.4e6
    return c


def test_spectrum_shape(cfg):
    iq = SimulatedBackend(100e6, 2.4e6).read_iq(cfg.rf.nfft * 2)
    spec = power_spectrum_db(iq, cfg.rf.nfft, cfg.rf.sample_rate_hz)
    # Complex FFT returns nfft bins after fftshift (full [-rate/2, rate/2) range)
    assert spec.shape == (cfg.rf.nfft,)
    assert np.all(np.isfinite(spec))


def test_carrier_raises_peak(cfg):
    src = SimulatedBackend(100e6, 2.4e6, carrier_hz=500e3)
    quiet = averaged_spectrum([src.read_iq(1 << 16)], cfg.rf.nfft, cfg.rf.sample_rate_hz)
    src.inject_carrier = True
    loud = averaged_spectrum([src.read_iq(1 << 16)], cfg.rf.nfft, cfg.rf.sample_rate_hz)

    start = cfg.rf.center_hz - cfg.rf.sample_rate_hz / 2
    bw = cfg.rf.sample_rate_hz / cfg.rf.nfft
    _, fq = peak_in_band(loud - quiet, start, bw, 300e3, 800e3)
    assert abs(fq - (cfg.rf.center_hz + 500e3)) < bw * 4


def test_decimate_width():
    assert decimate_max(np.arange(100.0), 20).size == 20


def test_waterfall_bounded(cfg):
    wf = Waterfall(cfg.waterfall)
    for _ in range(200):
        db = np.random.uniform(-120, -10, cfg.rf.nfft // 2)
        valid = np.ones(cfg.rf.nfft // 2, dtype=bool)
        frame = type("F", (), {"db": db, "valid": valid})()
        wf.push(frame)
    rendered = wf.render()
    assert len(wf._rows) == cfg.waterfall.height
    assert "\n" in rendered.plain