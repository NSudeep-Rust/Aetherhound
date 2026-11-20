# Autoencoder baseline (before changes)

Ran tools/ae_vs_attack_check.py against the shipped ONNX model.

- AE trip threshold: 0.00453
- Clean false alarms: 0/468
- Attack patches caught by the autoencoder alone: 0/468
- Attack median score 0.00021 vs trip 0.00453

Detection in the demo comes from the tonal/keying guard, not the autoencoder.

## Results table

| | Shipped model | My retrained control |
|---|---|---|
| AE trip | 0.00453 | 0.00316 |
| Attack median | 0.00021 | 0.00068 |
| Caught by AE alone | 0/468 | 0/468 |
| False alarms | 0/468 | 0/468 |

Control = retrained with 40 recordings, 20 epochs, default settings.