# AetherHound

Detects RF and acoustic covert channels in an air-gapped room by learning the normal ambient signature and alarming on machine-like transmission.

## Architecture

```
AetherHound/
├── capture/           # Audio & RF recording
│   ├── record_audio.py      # Legacy audio capture
│   ├── live_audio.py        # Live microphone callback queue
│   ├── rf_source.py         # RF backend abstraction (RTL-SDR, HackRF, simulated)
│   └── record_rf.py         # RF baseline calibration
├── dsp/               # Signal processing pipeline
│   ├── pipeline.py          # Audio: bandpass + notches + STFT
│   ├── patch_generator.py   # 64-frame patch slicing
│   └── rf_pipeline.py       # RF: IQ -> dB spectrum, decimation, peak search
├── train/             # Model training & calibration
│   ├── model.py             # AudioAutoencoder architecture
│   ├── run_training.py      # Training loop (15 epochs)
│   └── calibrate_threshold.py  # 99.5th percentile threshold
├── export/            # ONNX export
│   └── export_onnx.py
├── infer/             # Live detection engines
│   ├── detector.py          # Audio: ONNX runtime + hysteresis + SQLite
│   └── rf_detector.py       # RF: z-score baseline + peak freq + SQLite
├── ui/                # Terminal dashboard
│   ├── dashboard.py         # Legacy Rich Live split-screen UI
│   ├── waterfall.py         # Truecolour RF waterfall
│   └── alert.py             # Non-blocking buzzer thread
├── deploy/            # Production runner
│   └── run_app.py           # Live mic + RF, 30 FPS, waterfall, buzzer
├── tests/             # Pytest validation
│   ├── test_core.py         # Audio DSP + state machine
│   └── test_rf.py           # RF spectrum + waterfall
├── mock/              # Synthetic data generator
│   └── generate_signals.py
├── config.toml        # All hyperparameters (now actually parsed)
├── config_loader.py   # Typed config loader
├── requirements.txt   # Python dependencies
├── requirements-rf.txt  # SDR dependencies (pyrtlsdr, pyhackrf)
└── models/            # Trained artifacts (gitignored)
    ├── autoencoder.pt
    ├── autoencoder.onnx
    ├── threshold.txt
    └── rf_baseline.npz
```

## Quick Start

```bash
# 1. Setup environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-rf.txt   # optional: for real SDR hardware

# 2. Generate synthetic data (baseline + attack)
python mock/generate_signals.py

# 3. Build DSP patches (with filters now applied)
python dsp/pipeline.py
python dsp/patch_generator.py

# 4. Train autoencoder
python train/run_training.py

# 5. Calibrate threshold
python train/calibrate_threshold.py

# 6. Export ONNX
python export/export_onnx.py

# 7. Calibrate RF baseline (once per environment)
python capture/record_rf.py --seconds 30

# 8. Run live detection with dashboard (30 FPS, waterfall, buzzer)
python deploy/run_app.py
```

## Key Results

| Metric | Value |
|--------|-------|
| Training Loss (final) | 0.009942 |
| Threshold (99.5th pct) | 0.010697 |
| Attack Detection Rate | 133/468 patches (28.4%) |
| ONNX Parity (max diff) | 0.000000 |
| Test Suite | 4/4 passed (2 core + 2 RF) |

## Configuration

All parameters in `config.toml` (now actually parsed by `config_loader.py`):

- **Audio**: 48kHz, 1024 blocksize, device `-1` (system default)
- **Bandpass**: 20–22000 Hz (ultrasonic capable)
- **Notch filters**: 50 Hz base × 8 harmonics, Q=30
- **DSP**: nfft=512, hop=32, patch=64
- **Detection**: armed_frames=2, clear_frames=5, hold=1.5s, clear_ratio=0.7
- **RF**: enabled=true, backend=auto, 100 MHz center, 2.4 MS/s, gain=28 dB
- **Waterfall**: 108×26, -85 to -25 dB
- **UI**: 30 Hz refresh, buzzer @ 2200 Hz, 2s cooldown

## RF Layer

`capture/rf_source.py` supports RTL-SDR (`pyrtlsdr`), HackRF (`pyhackrf`), and a
simulated source for hardware-free work. `backend = "auto"` tries real hardware
first. Calibrate a baseline once with `capture/record_rf.py`, then
`infer/rf_detector.py` scores each spectrum as a per-bin z-score against it
and logs the peak frequency.

Limits, stated plainly: the waterfall covers one 2.4 MHz window, not wideband
sweep. Detection is single-antenna, so it identifies the *frequency* of a
transmission but not its *direction*. Rack-level localisation needs a second
receiver.

## Files

- `models/autoencoder.pt` - PyTorch model weights
- `models/autoencoder.onnx` - ONNX export for inference
- `models/threshold.txt` - Anomaly threshold (0.010697)
- `models/rf_baseline.npz` - RF per-bin mean/std baseline
- `events.db` - SQLite event log (audio + RF tables)
- `deploy/verified_receipt.txt` - Deployment verification

## License

MIT
