"""Production runner: swept RF + live audio, wide waterfall, bearing, buzzer."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from rich.console import Console
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
from dsp.pipeline import apply_filters, spectrogram_from_array
from infer.detector import AetherHoundEngine
from infer.rf_detector import RFAnomalyDetector
from ui.alert import Buzzer
from ui.waterfall import Waterfall

console = Console()


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    engine = AetherHoundEngine(args.config)
    wf = Waterfall(cfg.waterfall)
    buzzer = Buzzer(cfg.ui.buzzer_hz, cooldown_s=cfg.ui.buzzer_cooldown_s) if cfg.ui.buzzer else None

    rf_enabled = bool(getattr(cfg.rf, "enabled", True))
    baseline_path = Path(getattr(cfg.rf, "baseline_path", "models/rf_baseline.npz"))

    if rf_enabled and baseline_path.exists():
        src = open_rf_source(cfg.rf)
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

    audio = LiveAudioSource(cfg.audio)
    audio.start()

    dsp, frame_dt = cfg.dsp, 1.0 / max(cfg.ui.refresh_hz, 1)
    buf_len = dsp.nfft + (dsp.patch - 1) * dsp.hop
    buf = np.zeros(0, dtype=np.float32)
    audio_state, a_score = "CLEAN", 0.0
    last_bearing = 0.0

    try:
        with Live(build_layout(bearing_est is not None), refresh_per_second=cfg.ui.refresh_hz, screen=True) as live:
            while True:
                t0 = time.perf_counter()

                block = audio.read(timeout=frame_dt)
                if block is not None:
                    buf = np.concatenate([buf, block])
                if buf.size >= buf_len:
                    chunk = buf[-buf_len:]
                    buf = buf[-dsp.nfft:]
                    filt = apply_filters(chunk, cfg.audio.rate, cfg.band, zero_phase=False)
                    patch = spectrogram_from_array(filt, cfg.audio.rate, dsp.nfft, dsp.hop)
                    if patch.shape[0] >= dsp.patch:
                        a_score, audio_state = engine.process_patch(patch[:dsp.patch])

                if worker is not None:
                    frame, z, peak_hz, rf_state, sweeps = worker.snapshot()
                else:
                    frame, z, peak_hz, rf_state, sweeps = None, 0.0, 0.0, "DISABLED", 0
                threat = rf_state == "THREAT_DETECTED" or audio_state == "THREAT_DETECTED"

                layout = build_layout(bearing_est is not None)
                label = "THREAT DETECTED" if threat else "CLEAN"
                style = "bold white on red" if threat else "bold black on green"
                layout["header"].update(Panel(Text(label, justify="center", style=style)))

                at = Table(expand=True, show_header=False)
                at.add_column(style="cyan")
                at.add_column(style="magenta")
                at.add_row("state", audio_state)
                at.add_row("score", f"{a_score:.6f}")
                at.add_row("threshold", f"{engine.threshold:.6f}")
                layout["audio"].update(Panel(at, title="Acoustic", border_style="cyan"))

                rf_text = Text()
                if frame is not None:
                    span = frame.span_hz
                    col = int((peak_hz - frame.freqs[0]) / span * wf.width) if span else None
                    wf.push(frame)
                    rf_text.append(
                        f"{peak_hz/1e6:.4f} MHz  z={z:.2f}  sweep #{sweeps}  {rf_state}\n",
                        style="bold white on red" if threat else "bold white",
                    )
                    rf_text.append_text(wf.render(col, frame.band_names))
                else:
                    rf_text.append("first sweep…", style="dim")
                layout["rf"].update(
                    Panel(rf_text, title="Wideband waterfall",
                          border_style="red" if threat else "blue")
                )

                if bearing_est is not None:
                    if peak_hz > 0 and time.time() - last_bearing > 5:
                        bearing_est.measure(peak_hz)
                        last_bearing = time.time()
                    bt = Text("\n".join(polar_plot(
                        bearing_est.last_deg, bearing_est.last_conf, cfg.bearing.positions)))
                    layout["bearing"].update(Panel(bt, title="Bearing (RSSI)", border_style="magenta"))

                layout["footer"].update(Panel(Text(
                    f"{cfg.ui.refresh_hz} fps UI  |  sweep ~{len(sweep_engine.centers) if sweep_engine else 0} steps"
                    f"  |  {len(cfg.sweep.band_plan)} bands"
                )))
                live.update(layout)

                if buzzer:
                    buzzer.set(threat)

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
    Path("deploy/verified_receipt.txt").write_text(
        "=== AETHERHOUND DEPLOYMENT RECEIPT ===\n"
        f"timestamp: {time.time()}\n"
        f"bands: {cfg.sweep.band_plan}\n"
        f"steps: {len(sweep_engine.centers)}\n"
        f"sweeps_completed: {worker.sweeps}\n"
        f"rf_backend: {src.name}\n"
    )
    console.print("[+] Receipt written", style="bold green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
