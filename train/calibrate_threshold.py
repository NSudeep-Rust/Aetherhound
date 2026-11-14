"""Calibrate operational thresholds on a held-out normal-data split."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn

from config_loader import load_config
from dsp.features import tonal_prominence
from dsp.normalization import FEATURE_TRANSFORM, load_reference_scale, normalise_spectrogram
from train.model import AudioAutoencoder
from train.run_training import split_indices


def _scores(model: AudioAutoencoder, patches: np.ndarray, reference_scale: float) -> np.ndarray:
    criterion = nn.MSELoss(reduction="none")
    results: list[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(patches), 64):
            features = torch.from_numpy(normalise_spectrogram(patches[start:start + 64], reference_scale))
            results.extend(criterion(model(features), features).mean(dim=(1, 2)).tolist())
    return np.asarray(results, dtype=np.float64)


def _tones(patches: np.ndarray, cfg) -> np.ndarray:
    return np.asarray([
        tonal_prominence(
            patch,
            cfg.audio.rate,
            cfg.dsp.nfft,
            min_hz=cfg.detect.tone_min_hz,
            max_hz=cfg.detect.tone_max_hz,
            window_bins=cfg.detect.tone_window_bins,
        )[0]
        for patch in patches
    ], dtype=np.float64)


def calibrate_and_validate() -> None:
    cfg = load_config()
    reference_scale = load_reference_scale()
    manifest_path = Path("models/training_manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError("missing models/training_manifest.json; run train/run_training.py first")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("feature_transform") != FEATURE_TRANSFORM:
        raise ValueError("training manifest feature transform does not match this runtime")

    model = AudioAutoencoder()
    model.load_state_dict(torch.load("models/autoencoder.pt", map_location="cpu", weights_only=True))
    baseline_data = np.load("data/baseline_patches.npy")
    _, validation_indices = split_indices(
        len(baseline_data), float(manifest["validation_fraction"]), int(manifest["seed"])
    )
    validation_data = baseline_data[validation_indices]

    baseline_errors = _scores(model, validation_data, reference_scale)
    baseline_tones = _tones(validation_data, cfg)
    ae_trip = float(np.quantile(baseline_errors, cfg.detect.calibration_quantile)) * cfg.detect.calibration_headroom
    tone_trip = float(np.quantile(baseline_tones, cfg.detect.calibration_quantile)) * cfg.detect.calibration_headroom
    threshold = ae_trip / cfg.detect.threat_margin

    Path(cfg.detect.threshold_file).write_text(f"{threshold:.9f}\n")
    Path(cfg.detect.tone_threshold_file).write_text(f"{tone_trip:.9f}\n")
    Path("models/calibration.json").write_text(json.dumps({
        "source": "held_out_baseline_split",
        "validation_count": int(len(validation_data)),
        "quantile": cfg.detect.calibration_quantile,
        "headroom": cfg.detect.calibration_headroom,
        "ae_trip": ae_trip,
        "tone_trip_db": tone_trip,
    }, indent=2) + "\n")

    attack_data = np.load("data/test_patches.npy")
    attack_errors = _scores(model, attack_data, reference_scale)
    attack_tones = _tones(attack_data, cfg)
    detected = (attack_errors > ae_trip) | (attack_tones > tone_trip)

    print(f"[+] Held-out calibration: {len(validation_data)} clean patches")
    print(f"    AE trip: {ae_trip:.6f} | tone trip: {tone_trip:.3f} dB")
    print(f"    held-out false alerts: {int(np.count_nonzero((baseline_errors > ae_trip) | (baseline_tones > tone_trip)))}/{len(validation_data)}")
    print(f"    attack patches detected: {int(detected.sum())}/{len(attack_data)}")
    if not np.any(detected):
        raise RuntimeError("no attack patch breached either calibrated guard")


if __name__ == "__main__":
    calibrate_and_validate()
