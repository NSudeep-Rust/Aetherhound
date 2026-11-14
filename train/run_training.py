"""Train the compact audio autoencoder on a deterministic held-out split."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from dsp.normalization import FEATURE_TRANSFORM, normalise_spectrogram
from train.model import AudioAutoencoder


def split_indices(n: int, validation_fraction: float = 0.2, seed: int = 2026) -> tuple[np.ndarray, np.ndarray]:
    """Return reproducible training and held-out validation indices."""
    if n < 10:
        raise ValueError("need at least 10 baseline patches for a train/validation split")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    validation_count = max(1, int(round(n * validation_fraction)))
    return order[validation_count:], order[:validation_count]


def train_model(
    data_path: str,
    model_save_path: str,
    epochs: int = 20,
    batch_size: int = 64,
    ref_scale_path: str = "models/ref_scale.txt",
    validation_fraction: float = 0.2,
    seed: int = 2026,
) -> None:
    torch.manual_seed(seed)
    raw_data = np.load(data_path)
    train_indices, validation_indices = split_indices(len(raw_data), validation_fraction, seed)
    train_raw = raw_data[train_indices]
    validation_raw = raw_data[validation_indices]

    # Only training data chooses the global reference scale. Calibration and
    # runtime reuse it; no per-patch normalisation is ever applied.
    patch_maxima = np.maximum(train_raw.max(axis=(1, 2)), 1e-10)
    ref_scale = float(np.median(patch_maxima))
    os.makedirs(os.path.dirname(ref_scale_path), exist_ok=True)
    Path(ref_scale_path).write_text(f"{ref_scale:.12g}\n")
    print(f"[+] Reference scale from training split: {ref_scale:.6g}")

    train_tensor = torch.from_numpy(normalise_spectrogram(train_raw, ref_scale))
    validation_tensor = torch.from_numpy(normalise_spectrogram(validation_raw, ref_scale))
    loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)

    model = AudioAutoencoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(f"[*] Training convolutional autoencoder for {epochs} epochs ({len(train_tensor)} train / {len(validation_tensor)} validation patches)")

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for (inputs,) in loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs), inputs)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(inputs)
        model.eval()
        with torch.no_grad():
            validation_loss = criterion(model(validation_tensor), validation_tensor).item()
        print(f"    Epoch {epoch + 1:02d}/{epochs:02d} | train {total / len(train_tensor):.6f} | validation {validation_loss:.6f}")

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    Path("models/training_manifest.json").write_text(json.dumps({
        "data_path": str(data_path),
        "n_patches": int(len(raw_data)),
        "validation_fraction": validation_fraction,
        "seed": seed,
        "feature_transform": FEATURE_TRANSFORM,
        "reference_scale": ref_scale,
        "train_count": int(len(train_indices)),
        "validation_count": int(len(validation_indices)),
    }, indent=2) + "\n")
    print(f"[+] Model saved to {model_save_path}; held-out split recorded in models/training_manifest.json")


if __name__ == "__main__":
    train_model("data/baseline_patches.npy", "models/autoencoder.pt")
