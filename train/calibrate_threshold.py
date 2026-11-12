import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from train.model import AudioAutoencoder


def _norm(patch):
    """Level-invariant per-patch peak normalization (matches infer/detector.py)."""
    peak = float(np.percentile(patch, 99.9))
    return torch.tensor(patch / max(peak, 1e-12), dtype=torch.float32).unsqueeze(0)


def calibrate_and_validate():
    # 1. Initialize the model and load trained weights
    model = AudioAutoencoder()
    model.load_state_dict(torch.load("models/autoencoder.pt"))
    model.eval()
    criterion = nn.MSELoss(reduction='none') # Compute error per item, not the batch average

    # 2. Load clean baseline patches
    baseline_data = np.load("data/baseline_patches.npy")

    # 3. Compute baseline errors using per-patch peak normalization
    baseline_errors = []
    with torch.no_grad():
        for i in range(baseline_data.shape[0]):
            patch = baseline_data[i]
            norm_patch = _norm(patch)
            reconstructed = model(norm_patch)
            err = criterion(reconstructed, norm_patch).mean().item()
            baseline_errors.append(err)
    baseline_errors = np.array(baseline_errors)

    # Set the anomaly threshold at the 99.9th percentile with 1.5x deadband headroom
    threshold = float(np.percentile(baseline_errors, 99.9)) * 1.5

    os.makedirs("models", exist_ok=True)
    with open("models/threshold.txt", "w") as f:
        f.write(f"{threshold:.6f}")
    print(f"[+] Baseline calibration complete. Threshold set at: {threshold:.6f}")
    print(f"[+] margin: max_baseline={baseline_errors.max():.6f} threshold={threshold:.6f} "
          f"headroom={threshold / max(baseline_errors.max(), 1e-12):.2f}x")

    headroom = threshold / max(baseline_errors.max(), 1e-12)
    if headroom < 1.3:
        print(f"[!] WARNING: headroom {headroom:.2f}x is below 1.30x minimum. "
              f"Regenerate baseline with python mock/generate_signals.py")

    # 4. Load and validate infected malware patches
    attack_data = np.load("data/test_patches.npy")

    attack_errors = []
    with torch.no_grad():
        for i in range(attack_data.shape[0]):
            patch = attack_data[i]
            norm_patch = _norm(patch)
            reconstructed = model(norm_patch)
            err = criterion(reconstructed, norm_patch).mean().item()
            attack_errors.append(err)
    attack_errors = np.array(attack_errors)

    # Count how many infected blocks successfully breached our guardrail
    anomalies_detected = np.sum(attack_errors > threshold)
    total_patches = len(attack_errors)

    print(f"[*] Validating attack signatures against threshold...")
    print(f"    - Baseline Max Error: {baseline_errors.max():.6f}")
    print(f"    - Attack Max Error:   {attack_errors.max():.6f}")
    print(f"[+] Detection Results: Flagged {anomalies_detected} out of {total_patches} attack patches as MALWARE!")

    assert anomalies_detected > 0, "Validation failed: Malware pulses did not trigger the anomaly threshold!"
    print("[+] Anomaly detection threshold engine validated successfully!")

if __name__ == "__main__":
    calibrate_and_validate()
