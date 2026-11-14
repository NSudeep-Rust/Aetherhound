"""Create a separate real-room calibration profile without changing virtual demo assets."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config_loader import load_config
from dsp.streaming import StreamingPatchBuilder
from infer.detector import AetherHoundEngine


def _float_audio(samples: np.ndarray) -> np.ndarray:
    data = np.asarray(samples)
    if data.ndim > 1:
        data = data[:, 0]
    if np.issubdtype(data.dtype, np.integer):
        return data.astype(np.float32) / np.iinfo(data.dtype).max
    return data.astype(np.float32)


def load_wav(path: Path, expected_rate: int) -> np.ndarray:
    rate, samples = wavfile.read(path)
    if rate != expected_rate:
        raise ValueError(f"{path} is {rate} Hz; expected {expected_rate} Hz")
    return _float_audio(samples)


def record_microphone(seconds: float, cfg, device: int | None) -> np.ndarray:
    # This import is intentionally local: virtual-only runs never initialise an
    # operating-system audio device.
    import sounddevice as sd

    selected = None if device in (None, -1) else device
    print("[?] Keep the sounds you want treated as CLEAN present during this recording.")
    print("    For example: normal fan/room noise, speech, and a short sample of music.")
    print(f"[*] Recording {seconds:.0f} seconds from microphone {selected if selected is not None else 'default'}...")
    samples = sd.rec(
        int(seconds * cfg.audio.rate),
        samplerate=cfg.audio.rate,
        channels=cfg.audio.channels,
        dtype="float32",
        device=selected,
    )
    sd.wait()
    return _float_audio(samples)


def build_patches(samples: np.ndarray, cfg) -> list[np.ndarray]:
    """Use exactly the same causal filter and patch timing as the dashboard."""
    builder = StreamingPatchBuilder(cfg.audio, cfg.band, cfg.dsp, cfg.detect)
    patches: list[np.ndarray] = []
    blocksize = cfg.audio.blocksize
    for offset in range(0, samples.size, blocksize):
        block = samples[offset : offset + blocksize]
        if block.size < blocksize:
            block = np.pad(block, (0, blocksize - block.size))
        patch = builder.push(block)
        if patch is not None:
            patches.append(patch)
    return patches


def build_profile(patches: list[np.ndarray], engine: AetherHoundEngine, cfg, name: str) -> dict:
    if len(patches) < 20:
        raise ValueError("not enough audio to calibrate; record at least 5 seconds")
    scores = engine.score_many(patches)
    tones_and_freqs = [engine.tone_score(patch) for patch in patches]
    tones = np.array([item[0] for item in tones_and_freqs])
    rms = np.array([float(np.sqrt(np.mean(patch ** 2))) for patch in patches])

    # The profile is intentionally conservative: it learns the environment the
    # user called normal and keeps an additional safety margin above its maxima.
    ae_trip = float(scores.max()) * 1.15
    tone_trip_db = float(tones.max()) * 1.15
    silence_rms = max(float(cfg.detect.silence_rms), float(np.percentile(rms, 5)) * 0.20)
    return {
        "format": "aetherhound-room-profile-v1",
        "name": name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sample_rate": cfg.audio.rate,
        "nfft": cfg.dsp.nfft,
        "patches": len(patches),
        "ae_trip": ae_trip,
        "tone_trip_db": tone_trip_db,
        "silence_rms": silence_rms,
        "baseline_ae_max": float(scores.max()),
        "baseline_tone_max_db": float(tones.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=45.0, help="microphone baseline duration")
    parser.add_argument("--device", type=int, default=None, help="input-device number; default uses system input")
    parser.add_argument("--from-wav", type=Path, default=None, help="calibrate from a 48 kHz WAV instead of a microphone")
    parser.add_argument("--profile", type=Path, default=Path("models/profiles/room.json"))
    parser.add_argument("--name", default="room", help="display name for this profile")
    parser.add_argument("--save-wav", type=Path, default=None, help="optional copy of captured microphone audio")
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")

    cfg = load_config()
    if args.from_wav is not None:
        samples = load_wav(args.from_wav, cfg.audio.rate)
        source = str(args.from_wav)
    else:
        samples = record_microphone(args.seconds, cfg, args.device)
        source = "microphone"
        if args.save_wav is not None:
            args.save_wav.parent.mkdir(parents=True, exist_ok=True)
            wavfile.write(args.save_wav, cfg.audio.rate, samples)
            print(f"[+] Saved optional room recording: {args.save_wav}")

    patches = build_patches(samples, cfg)
    engine = AetherHoundEngine()
    profile = build_profile(patches, engine, cfg, args.name)
    profile["source"] = source
    args.profile.parent.mkdir(parents=True, exist_ok=True)
    args.profile.write_text(json.dumps(profile, indent=2) + "\n")

    print(f"[+] Room profile written: {args.profile}")
    print(f"    patches: {profile['patches']}")
    print(f"    AE trip: {profile['ae_trip']:.6f} (baseline max {profile['baseline_ae_max']:.6f})")
    print(f"    tone trip: {profile['tone_trip_db']:.2f} dB (baseline max {profile['baseline_tone_max_db']:.2f})")
    print("[+] Start the real-room dashboard with:")
    print(f"    python deploy/run_app.py --source microphone --profile {args.profile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
