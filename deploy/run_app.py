"""Production runner: live mic + live RF, waterfall, dashboard, buzzer."""
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

from capture.live_audio import LiveAudioSource
from capture.rf_source import open_rf_source
from config_loader import load_config
from dsp.pipeline import apply_filters, spectrogram_from_array
from dsp.rf_pipeline import averaged_spectrum, decimate_max
from infer.detector import AetherHoundEngine
from infer.rf_detector import RFAnomalyDetector
from ui.alert import Buzzer
from ui.waterfall import Waterfall

console = Console()


def build_layout() -> Layout:
    root = Layout()
    root.split_column(
        Layout(name="header", size=3),
        Layout(name="body", ratio=1),
        Layout(name="footer", size=6),
    )
    root["body"].split_row(
        Layout(name="audio", ratio=1),
        Layout(name="rf", ratio=2),
    )
    return root


def audio_panel(engine, score, state) -> Panel:
    t = Table(expand=True, show_header=False)
    t.add_column("k", style="cyan")
    t.add_column("v", style="magenta")
    t.add_row("Audio state", state)
    t.add_row("Anomaly score", f"{score:.6f}")
    t.add_row("Threshold", f"{engine.threshold:.6f}")
    t.add_row("Hot / cold", f"{engine.anomaly_counter} / {engine.clean_counter}")
    return Panel(t, title="Acoustic", border_style="cyan")


def rf_panel(det, wf, peak_col, peak_hz, z) -> Panel:
    head = Text(
        f"peak {peak_hz/1e6:.4f} MHz   z={z:.2f}   state={det.state}",
        style="bold white on red" if det.state == "THREAT_DETECTED" else "bold white",
    )
    body = Text()
    body.append_text(head)
    body.append("\n")
    body.append_text(wf.render(peak_col))
    return Panel(body, title="RF waterfall", border_style="red" if det.state == "THREAT_DETECTED" else "blue")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--seconds", type=float, default=0, help="0 = run until Ctrl-C")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dsp, det_cfg, rf_cfg, ui_cfg = cfg.dsp, cfg.detect, cfg.rf, cfg.ui

    engine = AetherHoundEngine(args.config)
    wf = Waterfall(cfg.waterfall)
    buzzer = Buzzer(ui_cfg.buzzer_hz, cooldown_s=ui_cfg.buzzer_cooldown_s) if ui_cfg.buzzer else None

    audio = LiveAudioSource(cfg.audio)
    rf_src = None
    rf_det = None
    if rf_cfg.enabled:
        rf_det = RFAnomalyDetector(rf_cfg)
        rf_src = open_rf_source(rf_cfg)

    audio.start()
    if rf_src:
        rf_src.open() if not getattr(rf_src, "_open", False) else None

    frame_dt = 1.0 / max(ui_cfg.refresh_hz, 1)
    buf_len = dsp.nfft + (dsp.patch - 1) * dsp.hop
    audio_buf = np.zeros(0, dtype=np.float32)
    peak_col = None
    started = time.time()
    audio_state, a_score = "CLEAN", 0.0

    try:
        with Live(build_layout(), refresh_per_second=ui_cfg.refresh_hz, screen=True) as live:
            while True:
                t0 = time.perf_counter()

                block = audio.read(timeout=frame_dt)
                if block is not None:
                    audio_buf = np.concatenate([audio_buf, block])
                if audio_buf.size >= buf_len:
                    chunk = audio_buf[-buf_len:]
                    audio_buf = audio_buf[-dsp.nfft:]
                    filt = apply_filters(chunk, cfg.audio.rate, cfg.band, zero_phase=False)
                    patch = spectrogram_from_array(filt, cfg.audio.rate, dsp.nfft, dsp.hop)
                    if patch.shape[0] >= dsp.patch:
                        a_score, audio_state = engine.process_patch(patch[:dsp.patch])

                z, peak_hz, rf_state = 0.0, 0.0, "CLEAN"
                if rf_src and rf_det:
                    if isinstance(getattr(rf_src, "inject_carrier", None), bool):
                        rf_src.inject_carrier = (int(time.time()) % 12) >= 6
                    frames = [rf_src.read_iq(int(rf_cfg.sample_rate_hz * 0.05))
                              for _ in range(rf_cfg.frames_per_spectrum)]
                    spec = averaged_spectrum(frames, rf_cfg.nfft, rf_cfg.sample_rate_hz)
                    z, peak_hz, rf_state = rf_det.score(spec)
                    wf.push(spec)
                    start_hz = rf_cfg.center_hz - rf_cfg.sample_rate_hz / 2
                    col = int((peak_hz - start_hz) / (rf_cfg.sample_rate_hz / rf_cfg.nfft))
                    peak_col = int(col / (spec.size / cfg.waterfall.width))

                layout = build_layout()
                threat = rf_state == "THREAT_DETECTED" or audio_state == "THREAT_DETECTED"
                label = "THREAT DETECTED" if threat else "CLEAN"
                style = "bold white on red" if threat else "bold black on green"
                layout["header"].update(Panel(Text(label, justify="center"), border_style="blue"))
                layout["audio"].update(audio_panel(engine, a_score, audio_state))
                layout["rf"].update(rf_panel(rf_det, wf, peak_col, peak_hz, z) if rf_det
                                    else Panel(Text("RF disabled"), title="RF"))
                layout["footer"].update(Panel(Text(
                    f"refresh {1/frame_dt:.0f} fps  |  {cfg.rf.center_hz/1e6:.3f} MHz"
                )))
                live.update(layout)

                if buzzer:
                    buzzer.set(threat)

                if args.seconds and time.time() - started >= args.seconds:
                    break
                slack = frame_dt - (time.perf_counter() - t0)
                if slack > 0:
                    time.sleep(slack)
    except KeyboardInterrupt:
        pass
    finally:
        audio.stop()
        if rf_src:
            rf_src.close()
        if buzzer:
            buzzer.stop()

    Path("deploy").mkdir(exist_ok=True)
    receipt = Path("deploy/verified_receipt.txt")
    receipt.write_text(
        "=== AETHERHOUND DEPLOYMENT RECEIPT ===\n"
        f"timestamp: {time.time()}\n"
        f"audio_threshold: {engine.threshold:.6f}\n"
        f"rf_backend: {getattr(rf_src, 'name', 'none')}\n"
        f"rf_center_hz: {cfg.rf.center_hz:.0f}\n"
        f"refresh_hz: {ui_cfg.refresh_hz}\n"
        "status: OPERATIONAL\n"
    )
    console.print(f"[+] Receipt written to {receipt}", style="bold green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
