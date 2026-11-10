import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from train.model import AudioAutoencoder

def calibrate_and_validate():
    # 1. Initialize the model and load trained weights
    model = AudioAutoencoder()
    model.load_state_dict(torch.load("models/autoencoder.pt"))
    model.eval()
    criterion = nn.MSELoss(reduction='none') # Compute error per item, not the batch average
    
    # 2. Load clean baseline patches
    baseline_data = np.load("data/baseline_patches.npy")
    
    # 3. Compute per-patch maxima for fixed-reference normalization
    #    ref_scale = mean of baseline patch maxima (used by engine at inference)
    patch_maxima = []
    for i in range(baseline_data.shape[0]):
        patch = baseline_data[i]
        pmax = patch.max() if patch.max() > 0 else 1.0
        patch_maxima.append(pmax)
    ref_scale = float(np.mean(patch_maxima))
    
    # 4. Compute baseline errors using fixed-reference normalization
    baseline_errors = []
    with torch.no_grad():
        for i in range(baseline_data.shape[0]):
            patch = baseline_data[i]
            norm_patch = torch.tensor(patch / ref_scale, dtype=torch.float32).unsqueeze(0)
            reconstructed = model(norm_patch)
            err = criterion(reconstructed, norm_patch).mean().item()
            baseline_errors.append(err)
    baseline_errors = np.array(baseline_errors)
    
    # Set the anomaly threshold at the 99.5th percentile of normal room sound
    threshold = np.percentile(baseline_errors, 99.5)
    
    os.makedirs("models", exist_ok=True)
    with open("models/threshold.txt", "w") as f:
        f.write(f"{threshold:.6f}")
    with open("models/ref_scale.txt", "w") as f:
        f.write(f"{ref_scale:.10f}")
    print(f"[+] Baseline calibration complete. 99.5th Percentile Threshold set at: {threshold:.6f}")
    print(f"[+] Fixed reference scale (ref_scale) saved: {ref_scale:.10f}")
    
    # 5. Load and validate infected malware patches
    attack_data = np.load("data/test_patches.npy")
    
    attack_errors = []
    with torch.no_grad():
        for i in range(attack_data.shape[0]):
            patch = attack_data[i]
            norm_patch = torch.tensor(patch / ref_scale, dtype=torch.float32).unsqueeze(0)
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
