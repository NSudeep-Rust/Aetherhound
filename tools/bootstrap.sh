#!/usr/bin/env bash
# Build the data/ fixtures a fresh clone needs. data/ is gitignored (it's all
# synthetic and regenerable), so without this step `pytest -q` and
# `deploy/run_app.py` both fail looking for files that were never committed.
#
# This produces data/baseline/normal_room.wav, data/test/infected_attack.wav,
# data/baseline_patches.npy and data/test_patches.npy -- the same files
# train/run_training.py, train/calibrate_threshold.py and the test suite
# expect. It does NOT run training itself (that needs
# requirements-train.txt / torch); see the printed next steps below.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[*] Generating synthetic baseline/attack audio (mock/generate_signals.py)..."
python3 mock/generate_signals.py

echo "[*] Slicing audio into spectrogram patches (dsp/patch_generator.py)..."
python3 dsp/patch_generator.py

echo "[+] data/ is ready:"
echo "      data/baseline/normal_room.wav"
echo "      data/test/infected_attack.wav"
echo "      data/baseline_patches.npy"
echo "      data/test_patches.npy"
echo
echo "[*] To get a trained model and calibrated thresholds, also run:"
echo "      pip install -r requirements-train.txt"
echo "      python train/run_training.py"
echo "      python train/calibrate_threshold.py"
echo "      python export/export_onnx.py"
