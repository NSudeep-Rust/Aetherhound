# AetherHound

Detects RF and acoustic covert channels in an air-gapped room by learning the normal ambient signature and alarming on machine-like transmission.

## Architecture

```
AetherHound/
├── capture/           # Audio recording (Phase 2)
├── dsp/               # Signal processing pipeline (Phase 4)
│   ├── pipeline.py    # STFT spectrogram extraction
│   └── patch_generator.py  # 64-frame patch slicing
├── train/             # Model training & calibration (Phase 5-6)
│   ├── model.py       # AudioAutoencoder architecture
│   ├── run_training.py    # Training loop (15 epochs)
│   └── calibrate_threshold.py  # 99.5th percentile threshold
├── export/            # ONNX export (Phase 6.5)
│   └── export_onnx.py
├── infer/             # Live detection engine (Phase 8)
│   └── detector.py    # ONNX runtime + hysteresis + SQLite
├── ui/                # Terminal dashboard (Phase 9)
│   └── dashboard.py   # Rich Live split-screen UI
├── deploy/            # Production runner (Phase 11)
│   └── run_app.py
├── tests/             # Pytest validation (Phase 10)
│   └── test_core.py
├── mock/              # Synthetic data generator (Phase 1)
│   └── generate_signals.py
├── config.toml        # All hyperparameters
├── requirements.txt   # Python dependencies
└── models/            # Trained artifacts (gitignored)
    ├── autoencoder.pt
    ├── autoencoder.onnx
    └── threshold.txt
```

## Quick Start

```bash
# 1. Setup environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Generate synthetic data (baseline + attack)
python mock/generate_signals.py

# 3. Build DSP patches
python dsp/pipeline.py
python dsp/patch_generator.py

# 4. Train autoencoder
python train/run_training.py

# 5. Calibrate threshold
python train/calibrate_threshold.py

# 6. Export ONNX
python export/export_onnx.py

# 7. Run live detection with dashboard
python deploy/run_app.py
```

## Key Results

| Metric | Value |
|--------|-------|
| Training Loss (final) | 0.009942 |
| Threshold (99.5th pct) | 0.010697 |
| Attack Detection Rate | 133/468 patches (28.4%) |
| ONNX Parity (max diff) | 0.000000 |
| Test Suite | 2/2 passed |

## Configuration

All parameters in `config.toml`:
- Audio: 48kHz, 1024 blocksize, device index 4
- Bandpass: 300-8000 Hz
- Notch filters: 50, 100, 150, 200 Hz
- DSP: nfft=512, hop=32, patch=64
- Detection: armed_frames=2, clear_frames=5, hold=1.5s, clear_ratio=0.7

## Files

- `models/autoencoder.pt` - PyTorch model weights
- `models/autoencoder.onnx` - ONNX export for inference
- `models/threshold.txt` - Anomaly threshold (0.010697)
- `events.db` - SQLite event log
- `deploy/verified_receipt.txt` - Deployment verification

## License

MIT
