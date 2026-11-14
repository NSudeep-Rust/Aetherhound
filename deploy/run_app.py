"""Production runner: swept RF + live audio, wide waterfall, bearing, buzzer."""
from __future__ import annotations

import argparse
import sys
import time
from contextlib import nullcontext
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
from capture.simulated_audio import SimulatedAudioSource
from capture.rf_source import open_rf_source
from capture.sweep import SweepEngine, SweepWorker
from config_loader import load_config
from dsp.streaming import StreamingPatchBuilder
from infer.detector import AetherHoundEngine
from infer.rf_detector import RFAnomalyDetector
from ui.waterfall import Waterfall

console = Console()


def run_self_test() -> int:
    """Check clean and attack fixtures using the same calibrated runtime path."""
    engine = AetherHoundEngine()
    baseline = np.load("data/baseline_patches.npy")
    threshold = engine.threshold
    margin = engine.threat_margin
    limit = threshold * margin

    all_clean = True
    for i in range(min(128, len(baseline))):
        score, state, _ = engine.process_patch(baseline[i])
        if (state == "THREAT_DETECTED" or score > limit
                or (engine.tone_threshold is not None and engine.last_tone_score > engine.tone_threshold)):
            all_clean = False
            print(f"  FAIL: patch {i} score={score:.6f} state={state} limit={limit:.6f}")

    attack_engine = AetherHoundEngine()
    attack = np.load("data/test_patches.npy")
    attack_states = [attack_engine.process_patch(patch)[1] for patch in attack[:128]]
    attack_tones = [attack_engine.tone_score(patch)[0] for patch in attack[:128]]
    attack_detected = "THREAT_DETECTED" in attack_states
    tone_detected = (attack_engine.tone_threshold is not None
                     and max(attack_tones, default=0.0) > attack_engine.tone_threshold)

    if all_clean and attack_detected and tone_detected:
        print("PASS: 128 baseline patches stay below both calibrated guards")
        print("PASS: saved attack fixture reaches THREAT_DETECTED")
        print("PASS: tonal guard localises a simulated covert carrier")
        return 0

    print("FAIL: self-test did not preserve clean baseline or detect virtual attack")
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
    ap.add_argument("--source", choices=("simulation", "microphone"), default="simulation",
                    help="Audio input mode (simulation is the hardware-free default)")
    ap.add_argument("--profile", default="virtual",
                    help="Calibration profile: virtual (default), room, or a profile JSON path")
    ap.add_argument("--scenario", choices=("baseline", "attack", "mixed"), default="baseline",
                    help="Built-in virtual signal used in simulation mode")
    ap.add_argument("--audio-file", type=Path, default=None,
                    help="Use this 48 kHz WAV file as the virtual microphone input")
    ap.add_argument("--duration", type=float, default=None,
                    help="Stop after this many seconds (useful for a repeatable demo)")
    ap.add_argument("--no-ui", action="store_true", help="Print state changes instead of opening the live dashboard")
    ap.add_argument("--simulated-rf", action="store_true", help="Add the virtual RF waterfall; never opens SDR hardware")
    ap.add_argument("--inject", action="store_true", help="Inject a test carrier into --simulated-rf")
    ap.add_argument("--buzzer", action="store_true", help="Enable the optional local software alarm tone")
    ap.add_argument("--write-receipt", action="store_true", help="Write a deployment receipt when the run ends")
    ap.add_argument("--list-devices", action="store_true", help="List physical audio inputs and exit")
    ap.add_argument("--self-test", action="store_true", help="Verify the trained detector against saved virtual fixtures")
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

    if args.duration is not None and args.duration <= 0:
        ap.error("--duration must be positive")

    cfg = load_config(args.config)
    if args.simulated_rf or args.inject:
        cfg.rf.enabled = True
        cfg.rf.backend = "simulated"
    engine = AetherHoundEngine(args.config, profile=args.profile)
    wf = Waterfall(cfg.waterfall)
    # Derive waterfall dB range from RF baseline if available
    bp = Path("models/rf_baseline.npz")
    if bp.exists():
        _b = np.load(bp)
        wf.set_range_from_baseline(float(np.mean(_b["mean_db"])), float(np.mean(_b["std_db"])))
    # Audio waterfall for the acoustic panel
    audio_wf = AudioWaterfall(width=cfg.waterfall.width, height=20, sample_rate=cfg.audio.rate, nfft=cfg.dsp.nfft)
    if args.buzzer:
        # Import only when explicitly requested: the virtual workflow must not
        # touch an operating-system audio device.
        from ui.alert import Buzzer
        buzzer = Buzzer(cfg.ui.buzzer_hz, cooldown_s=cfg.ui.buzzer_cooldown_s)
    else:
        buzzer = None

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

    if args.source == "simulation":
        if args.audio_file is not None:
            simulation_paths = [args.audio_file]
            simulation_label = f"file: {args.audio_file}"
        elif args.scenario == "baseline":
            simulation_paths = [Path("data/baseline/normal_room.wav")]
            simulation_label = "baseline"
        elif args.scenario == "attack":
            simulation_paths = [Path("data/test/infected_attack.wav")]
            simulation_label = "attack"
        else:
            simulation_paths = [
                Path("data/baseline/normal_room.wav"),
                Path("data/test/infected_attack.wav"),
            ]
            simulation_label = "mixed"
        audio = SimulatedAudioSource(cfg.audio, simulation_paths, loop=True)
        audio.start()
        print(f"[+] Virtual audio source: {simulation_label}")
    else:
        # Physical microphone support is deliberately opt-in; the standard
        # project workflow is fully simulated.
        if args.profile == "virtual":
            print("[!] Using the virtual profile with a microphone. Normal room sound may false-alert; "
                  "create a room profile with train/calibrate_room.py first.")
        from capture.live_audio import LiveAudioSource
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
    patch_builder = StreamingPatchBuilder(cfg.audio, cfg.band, cfg.dsp, cfg.detect)
    audio_state, a_score = "CLEAN", 0.0
    a_anomalous = False
    last_bearing = 0.0
    patch = None  # Initialize to avoid UnboundLocalError

    # Build layout ONCE outside the loop to prevent UI flickering
    layout = build_layout(bearing_est is not None)

    # Track previous values to avoid redundant updates (prevents flicker)
    _prev_audio_state = None
    _prev_rf_state = None
    _prev_peak_hz = None
    _prev_sweeps = None
    _prev_threat = None
    _prev_footer_text = None
    _cached_audio_table = None
    _prev_tone_score = None
    _prev_headless_state = None
    started_at = time.monotonic()

    try:
        live_context = (Live(layout, refresh_per_second=cfg.ui.refresh_hz, screen=True, auto_refresh=False)
                        if not args.no_ui else nullcontext(None))
        with live_context as live:
            while True:
                t0 = time.perf_counter()

                # Microphone callbacks produce ~47 blocks/s while the UI redraws
                # at 30 Hz. Drain all queued callback blocks so the DSP never
                # sees gaps created by a slow display loop.
                if hasattr(audio, "drain"):
                    blocks = audio.drain()
                    if not blocks:
                        first = audio.read(timeout=frame_dt)
                        blocks = [first] if first is not None else []
                else:
                    first = audio.read(timeout=frame_dt)
                    blocks = [first] if first is not None else []

                produced_patch = False
                for block in blocks:
                    next_patch = patch_builder.push(block)
                    if next_patch is None:
                        continue
                    produced_patch = True
                    patch = next_patch
                    a_score, audio_state, a_anomalous = engine.process_patch(patch)
                    audio_wf.push(patch)

                # Process an audio patch only when the shared streaming assembler
                # has enough data. The room-calibration command uses this exact
                # path, so its thresholds match the live dashboard.
                if not produced_patch:
                    audio_wf.push(None)

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

                # Rebuild audio table only when state changes (stable display, no flicker)
                if (_prev_audio_state != audio_state or _cached_audio_table is None
                        or _prev_tone_score is None
                        or abs(engine.last_tone_score - _prev_tone_score) >= 0.25):
                    at = Table(expand=True, show_header=False)
                    at.add_column(style="cyan")
                    at.add_column(style="magenta")
                    at.add_row("state", audio_state)
                    at.add_row("AE score", f"{a_score:.6f}")
                    at.add_row("AE trip", f"{engine.threshold * engine.threat_margin:.6f}")
                    at.add_row("tone prominence", f"{engine.last_tone_score:.2f} dB")
                    at.add_row("tone peak", f"{engine.last_tone_hz:.0f} Hz")
                    at.add_row("tone trip", (f"{engine.tone_threshold:.2f} dB"
                                               if engine.tone_threshold is not None else "disabled"))
                    at.add_row("evidence", engine.last_reason)
                    # Add RMS for debugging
                    if patch is not None and patch.size > 0:
                        rms = float(np.sqrt(np.mean(patch ** 2)))
                        at.add_row("rms", f"{rms:.2e}")
                    _cached_audio_table = at
                    _prev_audio_state = audio_state
                    _prev_tone_score = engine.last_tone_score

                # Always update audio panel so waterfall scrolls smoothly every frame
                audio_content = Group(_cached_audio_table, Text(""), audio_wf.render())
                audio_border = ("red" if audio_state in ("THREAT_DETECTED", "SENSOR_LOSS")
                                else "yellow" if audio_state == "ARMED"
                                else "dim" if audio_state == "NO_SIGNAL" else "cyan")
                layout["audio"].update(Panel(audio_content, title="Acoustic", border_style=audio_border))

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

                if bearing_est is not None:
                    if peak_hz > 0 and time.time() - last_bearing > 5:
                        bearing_est.measure(peak_hz)
                        last_bearing = time.time()
                    bt = Text("\n".join(polar_plot(
                        bearing_est.last_deg, bearing_est.last_conf, cfg.bearing.positions)))
                    layout["bearing"].update(Panel(bt, title="Bearing (RSSI)", border_style="magenta"))

                # Only update footer when info changes (prevents constant refresh)
                footer_text = (
                    f"{cfg.ui.refresh_hz} fps UI  |  sweep ~{len(sweep_engine.centers) if sweep_engine else 0} steps"
                    f"  |  {len(cfg.sweep.band_plan)} bands"
                    f"  |  audio drops {getattr(audio, 'queue_drop_count', 0)}"
                    f"  |  callback warnings {getattr(audio, 'callback_status_count', 0)}"
                )
                if _prev_footer_text != footer_text:
                    layout["footer"].update(Panel(Text(footer_text)))
                    _prev_footer_text = footer_text

                # Always refresh for smooth waterfall scrolling (no animation when silent)
                if live is not None:
                    live.refresh()
                elif _prev_headless_state != audio_state:
                    console.print(
                        f"[{audio_state}] AE={a_score:.6f} | tone={engine.last_tone_score:.2f} dB "
                        f"@ {engine.last_tone_hz:.0f} Hz | {engine.last_reason}"
                    )
                    _prev_headless_state = audio_state

                if buzzer:
                    buzzer.set(audio_threat or rf_threat)

                if args.duration is not None and time.monotonic() - started_at >= args.duration:
                    break

                slack = frame_dt - (time.perf_counter() - t0)
                if slack > 0:
                    time.sleep(slack)
    except KeyboardInterrupt:
        pass
    finally:
        if worker is not None:
            worker.stop()
        audio.stop()
        engine.close()
        if src is not None:
            src.close()
        if buzzer:
            buzzer.stop()

    steps = len(sweep_engine.centers) if sweep_engine is not None else 0
    sweeps_completed = worker.sweeps if worker is not None else 0
    rf_backend = src.name if src is not None else "N/A (audio-only)"
    if args.write_receipt:
        Path("deploy").mkdir(exist_ok=True)
        Path("deploy/verified_receipt.txt").write_text(
            "=== AETHERHOUND DEPLOYMENT RECEIPT ===\n"
            f"timestamp: {time.time()}\n"
            f"audio_source: {args.source}\n"
            f"calibration_profile: {engine.profile_name}\n"
            f"scenario: {args.scenario}\n"
            f"bands: {cfg.sweep.band_plan}\n"
            f"steps: {steps}\n"
            f"sweeps_completed: {sweeps_completed}\n"
            f"rf_backend: {rf_backend}\n"
        )
        console.print("[+] Receipt written", style="bold green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
