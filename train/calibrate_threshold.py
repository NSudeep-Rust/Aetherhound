import os
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
    max_val = baseline_data.max() if baseline_data.max() > 0 else 1.0
    norm_baseline = torch.tensor(baseline_data / max_val, dtype=torch.float32)
    
    # 3. Compute baseline errors to establish the threshold
    with torch.no_grad():
        reconstructed_baseline = model(norm_baseline)
        # Flatten time and frequency to get an error score for each individual patch
        baseline_errors = criterion(reconstructed_baseline, norm_baseline).mean(dim=(1, 2)).numpy()
        
    # Set the anomaly threshold at the 99.5th percentile of normal room sound
    threshold = np.percentile(baseline_errors, 99.5)
    
    os.makedirs("models", exist_ok=True)
    with open("models/threshold.txt", "w") as f:
        f.write(f"{threshold:.6f}")
    print(f"[+] Baseline calibration complete. 99.5th Percentile Threshold set at: {threshold:.6f}")
    
    # 4. Load and validate infected malware patches
    attack_data = np.load("data/test_patches.npy")
    norm_attack = torch.tensor(attack_data / max_val, dtype=torch.float32)
    
    with torch.no_grad():
        reconstructed_attack = model(norm_attack)
        attack_errors = criterion(reconstructed_attack, norm_attack).mean(dim=(1, 2)).numpy()
        
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
