"""Prove the Morse payload end to end: run the real detection pipeline over
data/test_patches.npy and decode the keyed message back out.

mock/generate_signals.py's infected=True branch now keys its 4 kHz covert
tone with real Morse timing (see dsp/morse.py) instead of a flat
on-every-other-second toggle. This script is the other half: it feeds the
*actual* calibrated AetherHoundEngine.tone_score() decision for every patch
in data/test_patches.npy into dsp.morse.decode_timing() and prints what
comes back out -- this is the genuine "generator sends it, detector decodes
it" round trip, not a synthetic timing-only check (see tests/test_morse.py
for that one, which doesn't need the audio/STFT pipeline or a trained model).

Needs the full runtime stack (scipy, onnxruntime) and a calibrated model, so
run it the same way as the rest of the project:
    ./tools/bootstrap.sh          # if data/test_patches.npy doesn't exist yet
    python tools/decode_morse_attack.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from config_loader import load_config
from dsp.morse import decode_timing, presence_to_segments
from infer.detector import AetherHoundEngine


def main() -> int:
    cfg = load_config()
    patches_path = Path("data/test_patches.npy")
    if not patches_path.exists():
        print(f"error: {patches_path} not found -- run ./tools/bootstrap.sh first", file=sys.stderr)
        return 1

    engine = AetherHoundEngine()
    if engine.tone_threshold is None:
        print("error: tone guard is disabled (no tone_threshold calibrated)", file=sys.stderr)
        return 1

    patches = np.load(patches_path, mmap_mode="r")
    patch_dt = cfg.dsp.patch * cfg.dsp.hop / cfg.audio.rate

    flags = []
    for patch in patches:
        prominence_db, _freq_hz = engine.tone_score(np.asarray(patch))
        flags.append(prominence_db > engine.tone_threshold)

    on_count = sum(flags)
    print(f"[*] {on_count}/{len(flags)} patches flagged 'tone present' "
          f"(trip {engine.tone_threshold:.2f} dB, patch duration {patch_dt * 1000:.1f} ms)")

    segments = presence_to_segments(flags, patch_dt)
    segments = [(on and d >= 2 * patch_dt, d) for on, d in segments]
    decoded = decode_timing(segments)
    print(f"[+] Decoded message: {decoded!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
