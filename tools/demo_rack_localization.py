"""Demo: pin the most recent THREAT_DETECTED event to a virtual server rack.

There's only one real microphone in this project (see
capture/simulated_audio.py / capture/live_audio.py), so there is no actual
multi-mic amplitude data to localize from. This demo shows the capability
the brief asks for -- "pinpoints the infected server rack" -- by taking the
most recent incident's logged anomaly score as a stand-in source amplitude,
simulating what a small virtual mic array *would* have observed if the
source were really at each candidate rack, adding mic noise, and running
capture.rack_localization.estimate_source_rack() on it.

This is honestly a demonstration of the localization math (see
capture/rack_localization.py's own `python -m capture.rack_localization`
self-check for a noise-free, ground-truth-known version of the same thing),
not a claim that the single-mic deployment can really locate a source in a
room. A genuine deployment would need an actual mic array and per-mic gain
calibration.

Usage:
    python tools/demo_rack_localization.py                  # most recent THREAT_DETECTED in events.db
    python tools/demo_rack_localization.py --rack 2          # pretend the source really is at rack-3 (0-based)
    python tools/demo_rack_localization.py --amplitude 1.4   # skip the DB, use this amplitude directly
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from capture.rack_localization import default_room, estimate_source_rack, simulate_mic_amplitudes


def _latest_threat_score(db_path: str) -> float | None:
    if not Path(db_path).exists():
        return None
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT score FROM security_events WHERE current_state = 'THREAT_DETECTED' "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return float(row[0]) if row else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default="events.db")
    parser.add_argument("--rack", type=int, default=None,
                        help="0-based index of the rack to simulate the source at (default: random)")
    parser.add_argument("--amplitude", type=float, default=None,
                        help="source amplitude to simulate (default: most recent THREAT_DETECTED score, or 1.0)")
    parser.add_argument("--noise-floor", type=float, default=0.02)
    parser.add_argument("--rack-count", type=int, default=4)
    parser.add_argument("--mic-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    layout = default_room(rack_count=args.rack_count, mic_count=args.mic_count)

    amplitude = args.amplitude
    source_note = "given via --amplitude"
    if amplitude is None:
        amplitude = _latest_threat_score(args.db)
        source_note = f"most recent THREAT_DETECTED score in {args.db}"
    if amplitude is None or amplitude <= 0:
        amplitude = 1.0
        source_note = "no THREAT_DETECTED events found; using a placeholder amplitude"

    true_rack = args.rack if args.rack is not None else int(rng.integers(0, len(layout.racks)))
    if not 0 <= true_rack < len(layout.racks):
        print(f"error: --rack must be in [0, {len(layout.racks) - 1}]", file=sys.stderr)
        return 2

    observed = simulate_mic_amplitudes(
        layout, true_rack, source_amplitude=amplitude, noise_floor=args.noise_floor, rng=rng,
    )
    guess_index, scores = estimate_source_rack(observed, layout)

    print(f"[*] Source amplitude: {amplitude:.6f} ({source_note})")
    print(f"[*] Simulated mic readings: {np.array2string(observed, precision=4)}")
    print("[*] Rack match scores (cosine similarity, 1.0 = perfect):")
    for name, score in scores.items():
        marker = " <-- best match" if name == layout.rack_names[guess_index] else ""
        print(f"      {name}: {score:.3f}{marker}")
    print(f"[+] Estimated source: {layout.rack_names[guess_index]}")
    if args.rack is not None:
        outcome = "CORRECT" if guess_index == true_rack else "WRONG"
        print(f"[i] Ground truth was {layout.rack_names[true_rack]} -- {outcome}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
