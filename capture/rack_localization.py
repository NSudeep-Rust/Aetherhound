"""Virtual rack localization: which rack is the infected one?

The project brief asks the system to "pinpoint the infected server rack";
the README has admitted this was never implemented. There's no real
multi-mic array in this project (it's a software-only, one-microphone
demo -- see capture/simulated_audio.py), so this does the same thing
capture/bearing.py already does for RF direction-finding: simulate several
virtual microphones at fixed positions in a room, model how a sound
source's amplitude falls off with distance to each one, and estimate which
of a handful of known rack positions the source is closest to from the
*ratios* of amplitude across mics.

Ratios (not raw amplitudes) are the right quantity to match on, because the
source's own loudness is unknown -- see infer/detector.py's comment about
never renormalising a patch by itself. Two sources at the same rack but
different volumes produce the same ratios across mics; a source at a
different rack does not.

This is a room/placement *estimate*, not a measurement from a real sensor
array. With simulated distances it is deterministic and typically near-
perfect (see the self-check in `python -m capture.rack_localization`); a
real deployment would need per-mic gain calibration and would be much
noisier.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Position:
    x: float
    y: float

    def distance_to(self, other: "Position") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass(frozen=True)
class RoomLayout:
    """Fixed positions of candidate racks and listening microphones.

    Distances are in metres; any consistent unit works since only ratios of
    distances ever matter here. The default layout is a small server room:
    4 racks along one wall, 3 mics spread across the room so no two mics
    see the same amplitude ratio across racks (collinear mics can't
    disambiguate racks that are symmetric about the line).
    """
    racks: tuple[Position, ...]
    mics: tuple[Position, ...]

    def __post_init__(self) -> None:
        if len(self.racks) < 2:
            raise ValueError("need at least 2 racks to localize between")
        if len(self.mics) < 3:
            raise ValueError("need at least 3 mics to disambiguate rack positions in 2D")

    @property
    def rack_names(self) -> list[str]:
        return [f"rack-{i + 1}" for i in range(len(self.racks))]


def default_room(rack_count: int = 4, mic_count: int = 3,
                  room_width: float = 8.0, room_depth: float = 4.0) -> RoomLayout:
    """A small rectangular server room: racks along the back wall, mics spread
    along the front, matching the brief's "simulate 4 racks and 3-4 virtual
    mics" suggestion."""
    if rack_count < 2:
        raise ValueError("rack_count must be at least 2")
    if mic_count < 3:
        raise ValueError("mic_count must be at least 3")
    rack_xs = np.linspace(0.5, room_width - 0.5, rack_count)
    racks = tuple(Position(float(x), room_depth) for x in rack_xs)
    mic_xs = np.linspace(0.5, room_width - 0.5, mic_count)
    mics = tuple(Position(float(x), 0.0) for x in mic_xs)
    return RoomLayout(racks=racks, mics=mics)


def expected_amplitude_ratios(layout: RoomLayout, rack_index: int,
                              distance_exponent: float = 1.0,
                              min_distance: float = 0.25) -> np.ndarray:
    """Relative per-mic amplitude for a unit-amplitude source at ``rack_index``.

    Free-field sound pressure falls off as 1/distance (distance_exponent=1);
    intensity/power falls off as 1/distance**2. ``min_distance`` avoids a
    singularity if a mic sits exactly on a rack.
    """
    rack = layout.racks[rack_index]
    distances = np.array([max(mic.distance_to(rack), min_distance) for mic in layout.mics])
    amplitudes = 1.0 / distances**distance_exponent
    return amplitudes / amplitudes.sum()


def simulate_mic_amplitudes(layout: RoomLayout, rack_index: int, source_amplitude: float = 1.0,
                            distance_exponent: float = 1.0, noise_floor: float = 0.0,
                            rng: np.random.Generator | None = None) -> np.ndarray:
    """Amplitude each mic would observe from a source of ``source_amplitude``
    at ``layout.racks[rack_index]``, optionally with additive Gaussian noise
    (``noise_floor`` is a standard deviation, representing mic self-noise /
    uncorrelated room noise independent of the source)."""
    rack = layout.racks[rack_index]
    distances = np.array([max(mic.distance_to(rack), 0.25) for mic in layout.mics])
    amplitudes = source_amplitude / distances**distance_exponent
    if noise_floor > 0:
        rng = rng or np.random.default_rng()
        amplitudes = amplitudes + rng.normal(0.0, noise_floor, size=amplitudes.shape)
    return np.maximum(amplitudes, 0.0)


def estimate_source_rack(observed_amplitudes: np.ndarray, layout: RoomLayout,
                         distance_exponent: float = 1.0) -> tuple[int, dict[str, float]]:
    """Return ``(best_rack_index, {rack_name: score})``.

    ``score`` is a similarity in [0, 1] (1 = perfect match) between the
    observed amplitude ratios and each candidate rack's expected ratios,
    via cosine similarity -- scale-invariant, so the source's unknown
    absolute loudness never matters, only how it splits across mics.
    """
    observed = np.asarray(observed_amplitudes, dtype=np.float64)
    if observed.shape != (len(layout.mics),):
        raise ValueError(f"expected {len(layout.mics)} mic amplitudes, got {observed.shape}")
    if not np.any(observed > 0):
        raise ValueError("all observed amplitudes are zero -- no signal to localize")

    observed_ratio = observed / observed.sum()
    scores: dict[str, float] = {}
    best_index, best_score = -1, -math.inf
    for i, name in enumerate(layout.rack_names):
        expected_ratio = expected_amplitude_ratios(layout, i, distance_exponent)
        cosine = float(
            np.dot(observed_ratio, expected_ratio)
            / (np.linalg.norm(observed_ratio) * np.linalg.norm(expected_ratio))
        )
        scores[name] = cosine
        if cosine > best_score:
            best_index, best_score = i, cosine
    return best_index, scores


if __name__ == "__main__":
    print("[*] Self-check: simulate a source at each rack in turn and recover it from amplitude ratios.")
    layout = default_room(rack_count=4, mic_count=3)
    rng = np.random.default_rng(2026)
    all_correct = True
    for true_index, name in enumerate(layout.rack_names):
        observed = simulate_mic_amplitudes(
            layout, true_index, source_amplitude=1.0, noise_floor=0.01, rng=rng
        )
        guess_index, scores = estimate_source_rack(observed, layout)
        ok = guess_index == true_index
        all_correct &= ok
        scores_str = ", ".join(f"{n}={s:.3f}" for n, s in scores.items())
        print(f"    true={name:<8} guess={layout.rack_names[guess_index]:<8} "
              f"{'OK' if ok else 'WRONG'}  ({scores_str})")
    print("[+] All racks correctly localized!" if all_correct else "[!] Localization missed at least one rack.")
