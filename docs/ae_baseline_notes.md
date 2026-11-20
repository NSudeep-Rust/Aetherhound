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

## Experiment A: narrower bottleneck (48 -> 8 channels)
Trip 0.01260, attack median 0.00189, attack max 0.00331, trip/median 6.6x
(control 4.6x). AE alone caught 0/468, false alarms 0/468. Not an improvement:
clean loud recordings scored high too, so the trip rose faster than attacks.
Calibration's 181/468 includes the tonal guard (AE OR tone), not AE alone.