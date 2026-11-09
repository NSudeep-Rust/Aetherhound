"""Record RF spectra for baseline calibration and save mean/std per bin."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capture.rf_source import open_rf_source
from config_loader import load_config
from dsp.rf_pipeline import averaged_spectrum


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config).rf
    out = Path(args.out or cfg.baseline_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    per_frame = int(cfg.sample_rate_hz * 0.05)  # 50 ms per spectrum

    with open_rf_source(cfg) as src:
        print(f"[*] Calibrating on {src.name} @ {cfg.center_hz/1e6:.3f} MHz for {args.seconds:.0f}s")
        print("[*] Keep the room quiet — no transmitters, no hands on the antenna.")

        frames: list[np.ndarray] = []
        count = 0
        target = int(args.seconds / 0.05)
        while count < target:
            frames.append(averaged_spectrum([src.read_iq(per_frame)], cfg.nfft, cfg.sample_rate_hz))
            count += 1
            if count % 40 == 0:
                print(f"    {count}/{target} spectra", end="\r")

    print()
    stack = np.vstack(frames)
    mean_db = stack.mean(axis=0)
    std_db = stack.std(axis=0)
    std_db = np.maximum(std_db, 0.5)  # floor: quiet bins must not become infinitely sensitive

    np.savez_compressed(out, mean_db=mean_db, std_db=std_db,
                        center_hz=cfg.center_hz, sample_rate_hz=cfg.sample_rate_hz,
                        nfft=cfg.nfft)
    print(f"[+] Baseline written: {out} ({stack.shape[0]} spectra, {mean_db.size} bins)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())