"""Calibrate a swept RF baseline (per-bin mean/std across the whole band plan)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capture.rf_source import open_rf_source
from capture.sweep import SweepEngine
from config_loader import load_config


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", type=int, default=20)
    ap.add_argument("--out", default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out or cfg.rf.baseline_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open_rf_source(cfg.rf) as src:
        engine = SweepEngine(src, cfg.sweep, cfg.rf)
        print(f"[*] Calibrating {args.sweeps} sweeps on {src.name}")
        print("[*] Antenna connected, room quiet, no transmitters powered.")
        rows = []
        for i in range(args.sweeps):
            frame = engine.sweep_once()
            if frame.valid.sum() < 100:
                print(f"[!] sweep {i}: only {frame.valid.sum()} valid bins")
            rows.append(frame.db)
            print(f"    sweep {i+1}/{args.sweeps}", end="\r")
        freqs, valid = frame.freqs, frame.valid

    print()
    stack = np.vstack(rows)
    mean_db = stack.mean(axis=0)
    std_db = np.maximum(stack.std(axis=0), 0.5)

    np.savez_compressed(out, mean_db=mean_db, std_db=std_db, valid=valid, freqs=freqs)
    print(f"[+] Baseline: {out} — {stack.shape[0]} sweeps, {int(valid.sum())} valid bins")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())