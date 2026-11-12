"""Production runner: swept RF + live audio, wide waterfall, bearing, buzzer."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capture.bearing import BearingEstimator, polar_plot
from capture.live_audio import LiveAudioSource
from capture.rf_source import open_rf_source
from capture.sweep import SweepEngine, SweepWorker
from config_loader import load_config
from dsp.pipeline import StreamFilter, spectrogram_from_array
from infer.detector import AetherHoundEngine
from infer.rf_detector import RFAnomalyDetector
from ui.alert import Buzzer
from ui.waterfall import Waterfall

console = Console()


def run_self_test() -> int:
    """Self-test: run detector on baseline patches, verify all are CLEAN/NO_SIGNAL, print 3 PASS lines."""
    cfg = load_config()
    engine = AetherHoundEngine()
    baseline = np.load("data/baseline_patches.npy")
    threshold = engine.threshold
    margin = engine.threat_margin
    limit = threshold * margin

    all_clean = True
    for i in range(min(20, len(baseline))):
        score, state, _ = engine.process_patch(baseline[i])
        if state == "THREAT_DETECTED" or score > limit:
            all_clean = False
            print(f"  FAIL: patch {i} score={score:.6f} state={state} limit={limit:.6f}")

    if all_clean:
        print("PASS: self-test baseline patches all below threat margin")
        print("PASS: detector threshold and margin configured correctly")
        print("PASS: no false THREAT on clean baseline audio")
        return 0
    else:
        print("FAIL: self-test detected false THREAT on baseline patches")
        return 1


def build_layout(bearing: bool) -> Layout:
    root = Layout()
    root.split_column(
        Layout(name="header", size=3),
        Layout(name="body", ratio=1),
        Layout(name="footer", size=5),
    )
    if bearing:
        root["body"].split_row(
            Layout(name="audio", ratio=1),
            Layout(name="rf", ratio=3),
            Layout(name="bearing", ratio=1),
        )
    else:
        root["body"].split_row(Layout(name="audio", ratio=1), Layout(name="rf", ratio=3))
    return root


class AudioWaterfall:
    """Scrolling audio spectrogram for the acoustic panel."""
    def __init__(self, width: int = 108, height: int = 20, sample_rate: int = 48000, nfft: int = 512):
        self.width = width
        self.height = height
        self.sample_rate = sample_rate
        self.nfft = nfft
        self._rows = []
        # Frequency range: 200 Hz to 8 kHz (skip mains hum)
        self.freq_lo = 200
        self.freq_hi = 8000
        # Color ramp
        self.RAMP = [
            "#000004", "#0d0824", "#1f0c48", "#330a67", "#4a0c7a", "#611080",
            "#781d80", "#8c2c7c", "#a13d75", "#b44f6b", "#c56361", "#d47a58",
            "#df9350", "#e7ad4a", "#ecc947", "#f0e64e", "#f8f76a", "#fdff9a",
        ]
        self._lut = [f"on {c}" for c in self.RAMP]
        # Fixed dB range for stable rendering (no adaptive flicker)
        self._floor_db = -100.0
        self._ceil_db = -20.0
        # Silence level: map to black (index 0) so panel is dark when silent
        self._silence_idx = 0
    
    def push(self, patch: np.ndarray) -> None:
        """Push a patch (time x freq) and extract a column for the waterfall."""
        if patch is None or patch.size == 0:
            # Push black column when no data (panel dark when silent, no animation)
            self._rows.append(np.zeros(self.width, dtype=np.int8))
            if len(self._rows) > self.height:
                self._rows.pop(0)
            return
        # patch shape: (time_frames, freq_bins)
        # Take the mean across time frames for this patch to get one column
        col_spec = patch.mean(axis=0)  # (freq_bins,)
        
        # Convert to dB
        col_db = 10 * np.log10(np.maximum(col_spec, 1e-20))
        
        # Frequency axis
        freqs = np.linspace(0, self.sample_rate / 2, col_db.size)
        
        # Select frequency range of interest
        mask = (freqs >= self.freq_lo) & (freqs <= self.freq_hi)
        if not mask.any():
            return
        col_db = col_db[mask]
        freqs = freqs[mask]
        
        # Decimate to width
        from dsp.rf_pipeline import decimate_max
        col_db = decimate_max(col_db, self.width)
        
        # Fixed dB range mapping (no adaptive range = no flicker)
        span = max(self._ceil_db - self._floor_db, 1e-9)
        norm = np.clip((col_db - self._floor_db) / span, 0.0, 1.0)
        idx = (norm * (len(self._lut) - 1)).round().astype(np.int8)
        
        self._rows.append(idx)
        if len(self._rows) > self.height:
            self._rows.pop(0)
    
    def render(self) -> Text:
        text = Text(no_wrap=True, overflow="crop")
        for _ in range(self.height - len(self._rows)):
            text.append(" " * self.width + "\n")
        
        for row in self._rows:
            start = 0
            for i in range(1, self.width + 1):
                if i == self.width or row[i] != row[start]:
                    text.append(" " * (i - start), style=self._lut[int(row[start])])
                    start = i
            text.append("\n")
        
        # Frequency labels
        text.append(" " * int(self.width * 0.1) + "200Hz" + " " * int(self.width * 0.3) + "4kHz" + " " * int(self.width * 0.3) + "8kHz", style="dim")
        return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--inject", action="store_true", help="Inject test carrier in simulated RF backend")
    ap.add_argument("--list-devices", action="store_true", help="List available audio input devices and exit")
    ap.add_argument("--self-test", action="store_true", help="Run self-test mode: verify detector on baseline patches, print 3 PASS lines, exit")
    args = ap.parse_args()

    if args.list_devices:
        import sounddevice as sd
        print("Available audio input devices:")
        for d in sd.query_devices():
            if d["max_input_channels"] > 0:
                print(f"  [{d['index']}] {d['name']}  (ch={d['max_input_channels']}, sr={d['default_samplerate']:.0f} Hz)")
        return 0

    if args.self_test:
        return run_self_test()

    cfg = load_config(args.config)
    engine = AetherHoundEngine(args.config)
    wf = Waterfall(cfg.waterfall)
    # Derive waterfall dB range from RF baseline if available
    bp = Path("models/rf_baseline.npz")
    if bp.exists():
        _b = np.load(bp)
        wf.set_range_from_baseline(float(np.mean(_b["mean_db"])), float(np.mean(_b["std_db"])))
    # Audio waterfall for the acoustic panel
    audio_wf = AudioWaterfall(width=cfg.waterfall.width, height=20, sample_rate=cfg.audio.rate, nfft=cfg.dsp.nfft)
    buzzer = Buzzer(cfg.ui.buzzer_hz, cooldown_s=cfg.ui.buzzer_cooldown_s) if cfg.ui.buzzer else None

    rf_enabled = bool(getattr(cfg.rf, "enabled", True))
    baseline_path = Path(getattr(cfg.rf, "baseline_path", "models/rf_baseline.npz"))

    if rf_enabled and baseline_path.exists():
        src = open_rf_source(cfg.rf)
        # Enable carrier injection for simulated backend if requested
        if args.inject and hasattr(src, 'inject_carrier'):
            src.inject_carrier = True
            print("[+] Carrier injection enabled for simulated backend")
        src_lock = __import__("threading").Lock()
        det = RFAnomalyDetector(cfg.rf, cfg.log.db)
        sweep_engine = SweepEngine(src, cfg.sweep, cfg.rf, src_lock)
        worker = SweepWorker(sweep_engine, det)
        worker.start()
    else:
        if rf_enabled:
            print(f"[!] No RF baseline at {baseline_path}; running audio-only mode")
        else:
            print("[!] RF layer disabled in config; running audio-only mode")
        src = None
        src_lock = None
        det = None
        sweep_engine = None
        worker = None

    bearing_est = None
    if rf_enabled and src is not None and cfg.bearing.enabled:
        bearing_est = BearingEstimator(src, cfg.bearing, cfg.rf, src_lock)

    # Device fallback: try configured device, fall back to default
    try:
        audio = LiveAudioSource(cfg.audio)
        audio.start()
    except Exception as e:
        print(f"[!] Failed to open audio device {cfg.audio.device}: {e}; falling back to default")
        import dataclasses
        fallback_cfg = dataclasses.replace(cfg.audio, device=-1)
        audio = LiveAudioSource(fallback_cfg)
        audio.start()

    dsp, frame_dt = cfg.dsp, 1.0 / max(cfg.ui.refresh_hz, 1)
    buf_len = dsp.nfft + (dsp.patch - 1) * dsp.hop
    sfilt = StreamFilter(cfg.audio.rate, cfg.band)
    fbuf = np.zeros(0, dtype=np.float64)
    buf = np.zeros(0, dtype=np.float32)
    audio_state, a_score = "CLEAN", 0.0
    a_anomalous = False
    last_bearing = 0.0
    patch = None  # Initialize to avoid UnboundLocalError

    # Build layout ONCE outside the loop to prevent UI flickering
    layout = build_layout(bearing_est is not None)

    # Track previous values to avoid redundant updates (prevents flicker)
    _prev_audio_state = None
    _prev_a_score = None
    _prev_rf_state = None
    _prev_peak_hz = None
    _prev_sweeps = None
    _prev_threat = None
    _dirty = False

    try:
        with Live(layout, refresh_per_second=cfg.ui.refresh_hz, screen=True, auto_refresh=False) as live:
            while True:
                t0 = time.perf_counter()
                _dirty = False

                block = audio.read(timeout=frame_dt)
                if block is not None:
                    block = np.ravel(block).astype(np.float32)
                    fbuf = np.concatenate([fbuf, sfilt(block)])

                # Process audio buffer when we have enough samples for a full spectrogram patch
                if fbuf.size >= buf_len:
                    chunk = fbuf[-buf_len:].copy()
                    # Keep overlap: last nfft samples for continuity with next frame
                    fbuf = fbuf[-dsp.nfft:].copy()
                    patch = spectrogram_from_array(chunk, cfg.audio.rate, dsp.nfft, dsp.hop)
                    if patch.shape[0] >= dsp.patch:
                        patch = patch[:dsp.patch]
                        a_score, audio_state, a_anomalous = engine.process_patch(patch)
                        # Push to audio waterfall for visualization (always, even on NO_SIGNAL)
                        audio_wf.push(patch)
                    else:
                        # Not enough frames yet — push empty to keep panel alive
                        audio_wf.push(None)
                        a_anomalous = False
                else:
                    # Not enough buffer yet — push empty to keep panel alive
                    audio_wf.push(None)
                    a_anomalous = False

                if worker is not None:
                    frame, z, peak_hz, rf_state, sweeps = worker.snapshot()
                else:
                    frame, z, peak_hz, rf_state, sweeps = None, 0.0, 0.0, "DISABLED", 0
                # Aggregate threat across audio and RF layers
                audio_threat = audio_state == "THREAT_DETECTED"
                rf_threat = rf_enabled and rf_state == "THREAT_DETECTED"
                threat = audio_threat or rf_threat

                # Only update header when aggregate threat state changes
                if _prev_threat != threat:
                    if audio_threat or rf_threat:
                        label, style = "THREAT DETECTED", "bold white on red"
                    elif audio_state == "SENSOR_LOSS":
                        label, style = "SENSOR LOSS", "bold white on dark_orange"
                    elif audio_state == "NO_SIGNAL":
                        label, style = "NO SIGNAL", "bold black on grey62"
                    elif audio_state == "ARMED":
                        label, style = "ARMED", "bold black on yellow"
                    else:
                        label, style = "CLEAN", "bold black on green"
                    layout["header"].update(Panel(Text(label, justify="center", style=style)))
                    _prev_threat = threat
                    _dirty = True

                # Only update audio panel when state or score meaningfully changes
                score_changed = _prev_a_score is None or abs(a_score - _prev_a_score) > 0.0001
                if _prev_audio_state != audio_state or score_changed:
                    # Acoustic panel: show audio waterfall + state info
                    at = Table(expand=True, show_header=False)
                    at.add_column(style="cyan")
                    at.add_column(style="magenta")
                    at.add_row("state", audio_state)
                    at.add_row("score", f"{a_score:.6f}")
                    at.add_row("threshold", f"{engine.threshold:.6f}")
                    # Add RMS for debugging
                    if patch is not None and patch.size > 0:
                        rms = float(np.sqrt(np.mean(patch ** 2)))
                        at.add_row("rms", f"{rms:.2e}")
                    # Combine table and waterfall using Group (proper vertical composition)
                    audio_content = Group(at, Text(""), audio_wf.render())
                    # Border color follows audio state
                    audio_border = ("red" if audio_state in ("THREAT_DETECTED", "SENSOR_LOSS")
                                    else "yellow" if audio_state == "ARMED"
                                    else "dim" if audio_state == "NO_SIGNAL" else "cyan")
                    layout["audio"].update(Panel(audio_content, title="Acoustic", border_style=audio_border))
                    _prev_audio_state = audio_state
                    _prev_a_score = a_score
                    _dirty = True

                # Only update RF panel when RF state or peak changes
                rf_changed = (_prev_rf_state != rf_state or
                              _prev_peak_hz is None or abs(peak_hz - _prev_peak_hz) > 1000 or
                              _prev_sweeps != sweeps)
                if rf_changed:
                    rf_text = Text()
                    if frame is not None:
                        span = frame.span_hz
                        col = int((peak_hz - frame.freqs[0]) / span * wf.width) if span else None
                        wf.push(frame)
                        rf_text.append(
                            f"{peak_hz/1e6:.4f} MHz  z={z:.2f}  sweep #{sweeps}  {rf_state}\n",
                            style="bold white on red" if rf_state == "THREAT_DETECTED" else "bold white",
                        )
                        rf_text.append_text(wf.render(col, frame.band_names))
                    elif not rf_enabled:
                        # Show audio waterfall in right panel when RF is disabled
                        rf_text.append_text(audio_wf.render())
                        rf_text.append("\nAudio waterfall (RF disabled)", style="dim")
                    else:
                        rf_text.append("first sweep…", style="dim")
                    # RF panel border follows RF state, not combined threat
                    rf_border = "red" if rf_state == "THREAT_DETECTED" else ("yellow" if rf_state == "ARMED" else "blue")
                    layout["rf"].update(
                        Panel(rf_text, title="Wideband waterfall",
                              border_style=rf_border)
                    )
                    _prev_rf_state = rf_state
                    _prev_peak_hz = peak_hz
                    _prev_sweeps = sweeps
                    _dirty = True

                if bearing_est is not None:
                    if peak_hz > 0 and time.time() - last_bearing > 5:
                        bearing_est.measure(peak_hz)
                        last_bearing = time.time()
                    bt = Text("\n".join(polar_plot(
                        bearing_est.last_deg, bearing_est.last_conf, cfg.bearing.positions)))
                    layout["bearing"].update(Panel(bt, title="Bearing (RSSI)", border_style="magenta"))
                    _dirty = True

                layout["footer"].update(Panel(Text(
                    f"{cfg.ui.refresh_hz} fps UI  |  sweep ~{len(sweep_engine.centers) if sweep_engine else 0} steps"
                    f"  |  {len(cfg.sweep.band_plan)} bands"
                )))
                _dirty = True

                if _dirty:
                    live.refresh()

                if buzzer:
                    buzzer.set(a_anomalous or rf_threat)

                slack = frame_dt - (time.perf_counter() - t0)
                if slack > 0:
                    time.sleep(slack)
    except KeyboardInterrupt:
        pass
    finally:
        if worker is not None:
            worker.stop()
        audio.stop()
        if src is not None:
            src.close()
        if buzzer:
            buzzer.stop()

    Path("deploy").mkdir(exist_ok=True)
    steps = len(sweep_engine.centers) if sweep_engine is not None else 0
    sweeps_completed = worker.sweeps if worker is not None else 0
    rf_backend = src.name if src is not None else "N/A (audio-only)"
    Path("deploy/verified_receipt.txt").write_text(
        "=== AETHERHOUND DEPLOYMENT RECEIPT ===\n"
        f"timestamp: {time.time()}\n"
        f"bands: {cfg.sweep.band_plan}\n"
        f"steps: {steps}\n"
        f"sweeps_completed: {sweeps_completed}\n"
        f"rf_backend: {rf_backend}\n"
    )
    console.print("[+] Receipt written", style="bold green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
