import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from train.model import AudioAutoencoder
from config_loader import load_config
from dsp.features import tonal_prominence


def _norm(patch: np.ndarray, reference_scale: float) -> torch.Tensor:
    """Use the same fixed reference scale used by training and live inference."""
    normalised = np.maximum(np.asarray(patch, dtype=np.float32), 0.0) / reference_scale
    return torch.from_numpy(normalised).unsqueeze(0)


def calibrate_and_validate():
    cfg = load_config()
    reference_scale = float(Path("models/ref_scale.txt").read_text().strip())
    if reference_scale <= 0:
        raise ValueError("models/ref_scale.txt must contain a positive value")

    # 1. Initialize the model and load trained weights
    model = AudioAutoencoder()
    model.load_state_dict(torch.load("models/autoencoder.pt"))
    model.eval()
    criterion = nn.MSELoss(reduction='none') # Compute error per item, not the batch average

    # 2. Load clean baseline patches
    baseline_data = np.load("data/baseline_patches.npy")

    # 3. Compute baseline errors with the training-time reference scale.
    baseline_errors = []
    with torch.no_grad():
        for i in range(baseline_data.shape[0]):
            patch = baseline_data[i]
            norm_patch = _norm(patch, reference_scale)
            reconstructed = model(norm_patch)
            err = criterion(reconstructed, norm_patch).mean().item()
            baseline_errors.append(err)
    baseline_errors = np.array(baseline_errors)

    # Keep the calibration and runtime semantics identical.  The detector applies
    # ``threshold * threat_margin`` at runtime, so store the pre-margin value
    # here.  A 10% margin above the observed maximum gives a clean virtual
    # baseline zero false alerts without masking a real pulse train.
    ae_trip = float(baseline_errors.max()) * 1.10
    threshold = ae_trip / float(getattr(cfg.detect, "threat_margin", 1.0))

    baseline_tones = np.array([
        tonal_prominence(
            patch, cfg.audio.rate, cfg.dsp.nfft,
            min_hz=cfg.detect.tone_min_hz,
            max_hz=cfg.detect.tone_max_hz,
            window_bins=cfg.detect.tone_window_bins,
        )[0]
        for patch in baseline_data
    ])
    tone_threshold = float(baseline_tones.max()) * 1.10

    os.makedirs("models", exist_ok=True)
    with open("models/threshold.txt", "w") as f:
        f.write(f"{threshold:.6f}")
    with open(cfg.detect.tone_threshold_file, "w") as f:
        f.write(f"{tone_threshold:.6f}")
    print(f"[+] Baseline calibration complete. AE trip: {ae_trip:.6f} "
          f"(stored {threshold:.6f} x margin {cfg.detect.threat_margin})")
    print(f"[+] Tonal prominence trip: {tone_threshold:.3f} dB")

    # 4. Load and validate infected malware patches
    attack_data = np.load("data/test_patches.npy")

    attack_errors = []
    with torch.no_grad():
        for i in range(attack_data.shape[0]):
            patch = attack_data[i]
            norm_patch = _norm(patch, reference_scale)
            reconstructed = model(norm_patch)
            err = criterion(reconstructed, norm_patch).mean().item()
            attack_errors.append(err)
    attack_errors = np.array(attack_errors)

    attack_tones = np.array([
        tonal_prominence(
            patch, cfg.audio.rate, cfg.dsp.nfft,
            min_hz=cfg.detect.tone_min_hz,
            max_hz=cfg.detect.tone_max_hz,
            window_bins=cfg.detect.tone_window_bins,
        )[0]
        for patch in attack_data
    ])

    # Count attacks detected by either independently calibrated signal check.
    anomalies_detected = np.sum((attack_errors > ae_trip) | (attack_tones > tone_threshold))
    total_patches = len(attack_errors)

    print(f"[*] Validating attack signatures against threshold...")
    print(f"    - Baseline Max Error: {baseline_errors.max():.6f}")
    print(f"    - Attack Max Error:   {attack_errors.max():.6f}")
    print(f"    - Baseline Max Tone:  {baseline_tones.max():.3f} dB")
    print(f"    - Attack Max Tone:    {attack_tones.max():.3f} dB")
    print(f"[+] Detection Results: Flagged {anomalies_detected} out of {total_patches} attack patches as MALWARE!")

    assert not np.any(baseline_errors > ae_trip), "AE calibration permits a baseline alert"
    assert not np.any(baseline_tones > tone_threshold), "tone calibration permits a baseline alert"
    assert anomalies_detected > 0, "Validation failed: Malware pulses did not trigger either detector!"
    print("[+] Anomaly detection threshold engine validated successfully!")

if __name__ == "__main__":
    calibrate_and_validate()
