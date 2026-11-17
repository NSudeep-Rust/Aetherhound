"""Real Morse timing: encode text to on/off durations, and decode it back.

The project review noted that mock/fan_morse.py (the real-hardware CPU/fan
demo) keys actual Morse timing, but the *virtual*/simulated attack exercised
by mock/generate_signals.py and the test suite just toggles a tone on for a
flat 0.5 s every other second -- recognisable as "a keyed tone" but not an
actual encoded message. This module is the shared timing logic for closing
that gap: the simulated attack can now key real Morse (e.g. "SOS"), and
anything that observes per-patch tone-presence (dsp/keying.py's
KeyingDetector, or just raw tone_score() calls) can decode it back to text
with the same code.

Standard ITU timing, in units of one "dot":
    dot (dit)                    = 1 unit
    dash (dah)                   = 3 units
    gap between symbols in a char = 1 unit
    gap between characters        = 3 units
    gap between words              = 7 units
"""
from __future__ import annotations

from typing import Iterable, Sequence

MORSE_CODE: dict[str, str] = {
    "A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".", "F": "..-.",
    "G": "--.", "H": "....", "I": "..", "J": ".---", "K": "-.-", "L": ".-..",
    "M": "--", "N": "-.", "O": "---", "P": ".--.", "Q": "--.-", "R": ".-.",
    "S": "...", "T": "-", "U": "..-", "V": "...-", "W": ".--", "X": "-..-",
    "Y": "-.--", "Z": "--..", "0": "-----", "1": ".----", "2": "..---",
    "3": "...--", "4": "....-", "5": ".....", "6": "-....", "7": "--...",
    "8": "---..", "9": "----.",
}
_REVERSE_MORSE: dict[str, str] = {v: k for k, v in MORSE_CODE.items()}

# Relative durations in dot-units.
_DOT_UNITS = 1.0
_DASH_UNITS = 3.0
_INTRA_CHAR_GAP_UNITS = 1.0
_INTER_CHAR_GAP_UNITS = 3.0
_INTER_WORD_GAP_UNITS = 7.0


def to_morse(text: str) -> str:
    """Letters/digits of ``text`` as space-separated Morse, words joined by '/'."""
    words = []
    for word in text.upper().split():
        words.append(" ".join(MORSE_CODE[c] for c in word if c in MORSE_CODE))
    return " / ".join(w for w in words if w)


def _merge_segments(segments: list[tuple[bool, float]]) -> list[tuple[bool, float]]:
    """Coalesce adjacent same-state segments (defensive; encode_timing should
    never actually produce these, but decode_timing's callers build segments
    from arbitrary per-patch data where it's easy to end up with runs)."""
    merged: list[tuple[bool, float]] = []
    for is_on, duration in segments:
        if merged and merged[-1][0] == is_on:
            merged[-1] = (is_on, merged[-1][1] + duration)
        else:
            merged.append((is_on, duration))
    return merged


def encode_timing(text: str, dot_s: float = 0.15, lead_in_s: float = 0.0) -> list[tuple[bool, float]]:
    """Return ``[(is_on, duration_s), ...]`` for keying ``text`` in real Morse timing.

    Only characters present in :data:`MORSE_CODE` are encoded; anything else
    (punctuation, unsupported characters) is silently dropped, matching
    :func:`to_morse`. Unknown/empty input produces an empty list.
    """
    segments: list[tuple[bool, float]] = []
    if lead_in_s > 0:
        segments.append((False, lead_in_s))

    words = [w for w in text.upper().split() if any(c in MORSE_CODE for c in w)]
    for word_idx, word in enumerate(words):
        chars = [c for c in word if c in MORSE_CODE]
        for char_idx, char in enumerate(chars):
            code = MORSE_CODE[char]
            for sym_idx, sym in enumerate(code):
                segments.append((True, (_DASH_UNITS if sym == "-" else _DOT_UNITS) * dot_s))
                if sym_idx != len(code) - 1:
                    segments.append((False, _INTRA_CHAR_GAP_UNITS * dot_s))
            if char_idx != len(chars) - 1:
                segments.append((False, _INTER_CHAR_GAP_UNITS * dot_s))
        if word_idx != len(words) - 1:
            segments.append((False, _INTER_WORD_GAP_UNITS * dot_s))

    return _merge_segments(segments)


def presence_to_segments(flags: Sequence[bool], patch_duration_s: float) -> list[tuple[bool, float]]:
    """Collapse a per-patch on/off presence sequence into run-length segments.

    This is the bridge from what the live pipeline actually observes --
    dsp.keying.KeyingDetector and infer.detector.AetherHoundEngine.tone_score
    both work one patch at a time -- to the segment list decode_timing()
    expects. ``patch_duration_s`` is the wall-clock duration one patch
    represents (e.g. cfg.dsp.patch * cfg.dsp.hop / cfg.audio.rate).
    """
    segments: list[tuple[bool, float]] = []
    for flag in flags:
        is_on = bool(flag)
        if segments and segments[-1][0] == is_on:
            segments[-1] = (is_on, segments[-1][1] + patch_duration_s)
        else:
            segments.append((is_on, patch_duration_s))
    return segments


def decode_timing(segments: Iterable[tuple[bool, float]], dot_s: float | None = None,
                   tolerance: float = 0.5) -> str:
    """Reconstruct text from ``[(is_on, duration_s), ...]`` on/off segments.

    If ``dot_s`` is not given, it is estimated as the shortest "on" segment
    (the standard assumption: nothing is keyed shorter than one dot). Gaps
    and marks are classified against the dot/dash/gap unit boundaries
    (1, 3, 7 units) using ``tolerance`` as the fractional slack around each
    boundary, so jitter from patch-quantised timing doesn't misclassify a
    mark or gap that lands a bit off the ideal ratio.
    """
    segments = _merge_segments([(bool(on), float(d)) for on, d in segments if d > 0])
    marks = [d for on, d in segments if on]
    if not marks:
        return ""
    if dot_s is None:
        dot_s = min(marks)
    if dot_s <= 0:
        raise ValueError("dot_s must be positive")

    def classify(duration: float, boundaries: list[tuple[float, str]]) -> str:
        units = duration / dot_s
        best_label = boundaries[0][1]
        best_dist = float("inf")
        for ref_units, label in boundaries:
            dist = abs(units - ref_units)
            if dist < best_dist:
                best_dist, best_label = dist, label
        return best_label

    mark_boundaries = [(_DOT_UNITS, "."), (_DASH_UNITS, "-")]
    gap_boundaries = [
        (_INTRA_CHAR_GAP_UNITS, "symbol"),
        (_INTER_CHAR_GAP_UNITS, "char"),
        (_INTER_WORD_GAP_UNITS, "word"),
    ]

    words: list[str] = []
    chars: list[str] = []
    current = ""
    for is_on, duration in segments:
        if is_on:
            current += classify(duration, mark_boundaries)
        else:
            kind = classify(duration, gap_boundaries)
            if kind == "symbol":
                continue  # still inside the same character
            if current:
                chars.append(_REVERSE_MORSE.get(current, "?"))
                current = ""
            if kind == "word":
                words.append("".join(chars))
                chars = []
    if current:
        chars.append(_REVERSE_MORSE.get(current, "?"))
    if chars:
        words.append("".join(chars))

    return " ".join(w for w in words if w)
