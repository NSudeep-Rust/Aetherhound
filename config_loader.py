"""Central config loader. Parses config.toml once and exposes typed sections."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent / "config.toml"


@dataclass
class AudioCfg:
    device: int = -1
    rate: int = 48000
    blocksize: int = 1024
    channels: int = 1


@dataclass
class BandCfg:
    lo_hz: float = 20.0
    hi_hz: float = 22000.0
    notch_base_hz: float = 50.0
    notch_harmonics: int = 8
    notch_q: float = 30.0


@dataclass
class DspCfg:
    nfft: int = 512
    hop: int = 32
    patch: int = 64


@dataclass
class DetectCfg:
    threshold_file: str = "models/threshold.txt"
    armed_frames: int = 2
    clear_frames: int = 5
    hold_s: float = 1.5
    clear_ratio: float = 0.7


@dataclass
class LogCfg:
    db: str = "events.db"


@dataclass
class RFCfg:
    enabled: bool = False
    backend: str = "auto"
    device_index: int = 0
    center_hz: float = 100e6
    sample_rate_hz: float = 2.4e6
    gain_db: float = 28.0
    gain_auto: bool = False
    nfft: int = 2048
    frames_per_spectrum: int = 4
    baseline_path: str = "models/rf_baseline.npz"
    z_threshold: float = 4.5
    min_bins_over: int = 3
    peak_search_lo_hz: float = 300e3
    peak_search_hi_hz: float = 800e3
    rf_armed_frames: int = 3
    rf_clear_frames: int = 15


@dataclass
class WaterfallCfg:
    width: int = 108
    height: int = 26
    floor_db: float = -85.0
    ceil_db: float = -25.0


@dataclass
class UICfg:
    refresh_hz: int = 30
    buzzer: bool = True
    buzzer_hz: float = 2200.0
    buzzer_cooldown_s: float = 2.0


@dataclass
class SweepCfg:
    enabled: bool = True
    band_plan: list = field(default_factory=lambda: [[88e6, 108e6]])
    step_hz: float = 2e6
    dwell_ms: float = 15.0
    dc_guard_bins: int = 6


@dataclass
class BearingCfg:
    enabled: bool = False
    positions: list = field(default_factory=lambda: [0, 90, 180, 270])
    hold_s: float = 0.4
    target_bandwidth_hz: float = 100e3


@dataclass
class Config:
    audio: AudioCfg = field(default_factory=AudioCfg)
    band: BandCfg = field(default_factory=BandCfg)
    dsp: DspCfg = field(default_factory=DspCfg)
    detect: DetectCfg = field(default_factory=DetectCfg)
    log: LogCfg = field(default_factory=LogCfg)
    rf: RFCfg = field(default_factory=RFCfg)
    sweep: SweepCfg = field(default_factory=SweepCfg)
    bearing: BearingCfg = field(default_factory=BearingCfg)
    waterfall: WaterfallCfg = field(default_factory=WaterfallCfg)
    ui: UICfg = field(default_factory=UICfg)

    def validate(self) -> None:
        if self.band.lo_hz >= self.band.hi_hz:
            raise ValueError("band.lo_hz must be < band.hi_hz")
        nyquist = self.audio.rate / 2
        if self.band.hi_hz >= nyquist:
            raise ValueError(
                f"band.hi_hz {self.band.hi_hz} must be below Nyquist {nyquist}; "
                f"raise audio.rate"
            )
        if self.dsp.nfft % 2:
            raise ValueError("dsp.nfft must be even")
        if self.rf.enabled and self.rf.center_hz - self.rf.sample_rate_hz / 2 < 0:
            raise ValueError("rf window extends below 0 Hz")
        if self.rf.peak_search_lo_hz >= self.rf.peak_search_hi_hz:
            raise ValueError("rf peak search band inverted")
        if self.sweep.enabled:
            if self.sweep.step_hz > self.rf.sample_rate_hz:
                raise ValueError("sweep.step_hz must be <= rf.sample_rate_hz")
            for lo, hi in self.sweep.band_plan:
                if lo >= hi:
                    raise ValueError(f"sweep band inverted: {lo}-{hi}")


def _fill(obj, data: dict) -> None:
    for key, value in data.items():
        if hasattr(obj, key):
            setattr(obj, key, value)


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_PATH
    raw: dict = {}
    if path.exists():
        with path.open("rb") as fh:
            raw = tomllib.load(fh)

    cfg = Config()
    for section in ("audio", "band", "dsp", "detect", "log", "rf", "sweep", "bearing", "waterfall", "ui"):
        if section in raw and isinstance(raw[section], dict):
            _fill(getattr(cfg, section), raw[section])
    cfg.validate()
    return cfg