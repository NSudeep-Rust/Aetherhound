# Autoencoder baseline (before changes)

Ran tools/ae_vs_attack_check.py against the shipped ONNX model.

- AE trip threshold: 0.00453
- Clean false alarms: 0/468
- Attack patches caught by the autoencoder alone: 0/468
- Attack median score 0.00021 vs trip 0.00453

Detection in the demo comes from the tonal/keying guard, not the autoencoder.