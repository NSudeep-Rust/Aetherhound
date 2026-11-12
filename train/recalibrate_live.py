"""Fit threshold against a real recorded room baseline. No retraining."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from dsp.pipeline import extract_spectrogram
from infer.detector import AetherHoundEngine


def patches(path):
    spec = extract_spectrogram(path)
    return [spec[i:i + 64] for i in range(0, spec.shape[0] - 64, 32)]


def main():
    eng = AetherHoundEngine("config.toml")
    base = np.array([eng.score(p) for p in patches("data/baseline/live_room.wav")])
    atk = np.array([eng.score(p) for p in patches("data/test/infected_attack.wav")])

    b_rms = float(np.percentile([float(np.sqrt(np.mean(p ** 2))) for p in patches("data/baseline/live_room.wav")], 10))
    Path("models/baseline_rms.txt").write_text(f"{b_rms:.10f}")

    thr = max(float(np.percentile(base, 99.9)), float(base.max()) * 1.1) * 1.5
    Path("models/threshold.txt").write_text(f"{thr:.6f}")

    print(f"baseline  p50={np.percentile(base, 50):.6f} max={base.max():.6f}")
    print(f"attack    p50={np.percentile(atk, 50):.6f} max={atk.max():.6f}")
    print(f"threshold {thr:.6f}")
    print(f"separation {'OK' if atk.max() > thr else 'FAIL - retrain required'}")


if __name__ == "__main__":
    main()
