"""On/off keying detector: flags a tone only once it has toggled, not just appeared.

A covert acoustic (or RF) channel is almost always keyed -- switched on and
off -- to encode bits, so over time it looks like a short burst that repeats
several times at roughly one frequency. Two things that are NOT that, but
that a naive "is any tone above threshold" check cannot tell apart from it:

* A steady tone (fan hum, a mains harmonic the notch filters missed) is
  "on" for the entire observation window -- one long burst, never toggling.
* A single random spectral blip (part of the 75% noise floor randomly
  exceeding the trip for one patch) is "on" for one isolated patch -- one
  short burst, also never toggling.

Both produce exactly one burst. A real keyed carrier produces at least two,
separated by a genuine "off" gap, at a consistent frequency. Requiring at
least two bursts at (approximately) the same frequency is therefore enough
to reject both false-positive shapes while still catching the real signal.
"""
from __future__ import annotations

from collections import deque


class KeyingDetector:
    """Tracks per-patch tone presence/frequency and flags genuine keying.

    Feed it one decision per patch via :meth:`update`: whether a narrow-band
    tone was present (already thresholded by the caller, e.g. against a
    calibrated ``tone_threshold``) and, if so, at what frequency. It reports
    "keyed" only once at least ``min_bursts`` separate on-periods have been
    observed at a consistent frequency within the last ``window_patches``
    decisions.
    """

    def __init__(
        self,
        window_patches: int = 150,
        min_bursts: int = 2,
        freq_tolerance_hz: float = 200.0,
    ) -> None:
        if window_patches < 2:
            raise ValueError("window_patches must be at least 2")
        if min_bursts < 2:
            raise ValueError("min_bursts must be at least 2 -- a single burst never proves toggling")
        if freq_tolerance_hz < 0:
            raise ValueError("freq_tolerance_hz must not be negative")
        self.window_patches = window_patches
        self.min_bursts = min_bursts
        self.freq_tolerance_hz = freq_tolerance_hz
        self._present: deque[bool] = deque(maxlen=window_patches)
        self._freqs: deque[float] = deque(maxlen=window_patches)

    def reset(self) -> None:
        """Clear history, e.g. after a genuine silence gap breaks continuity."""
        self._present.clear()
        self._freqs.clear()

    def update(self, present: bool, frequency_hz: float) -> tuple[bool, float | None]:
        """Record one patch's tone-presence decision.

        Returns ``(is_keyed, frequency_hz)``: ``is_keyed`` is True once
        genuine on/off toggling has been observed, and ``frequency_hz`` is
        the representative frequency of the toggling channel (``None`` when
        not keyed).
        """
        self._present.append(bool(present))
        self._freqs.append(float(frequency_hz) if present else float("nan"))
        return self._evaluate()

    def _bursts(self) -> list[float]:
        """Representative (mean) frequency of each contiguous on-run so far."""
        bursts: list[float] = []
        current: list[float] = []
        for on, hz in zip(self._present, self._freqs):
            if on:
                current.append(hz)
            elif current:
                bursts.append(sum(current) / len(current))
                current = []
        if current:
            bursts.append(sum(current) / len(current))
        return bursts

    def _evaluate(self) -> tuple[bool, float | None]:
        bursts = self._bursts()
        if len(bursts) < self.min_bursts:
            return False, None

        # Cluster bursts by frequency: a channel toggling on/off repeats at
        # (roughly) one frequency rather than a new one each time, so
        # clustering keeps unrelated noise blips at different frequencies
        # from being miscounted as one toggling channel.
        clusters: list[list[float]] = []
        for hz in sorted(bursts):
            if clusters and hz - clusters[-1][-1] <= self.freq_tolerance_hz:
                clusters[-1].append(hz)
            else:
                clusters.append([hz])

        best = max(clusters, key=len)
        if len(best) >= self.min_bursts:
            return True, sum(best) / len(best)
        return False, None
