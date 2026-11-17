"""Unit tests for dsp/morse.py (encode/decode real Morse timing).

Pure numpy, no scipy/torch -- these exercise the generator's attack payload
(mock/generate_signals.py infected=True branch) and the decode path end to
end at the timing level, independent of the audio/STFT pipeline.
"""
from __future__ import annotations

import numpy as np
import pytest

from dsp.morse import decode_timing, encode_timing, presence_to_segments, to_morse


@pytest.mark.parametrize("message", ["SOS", "SOS AETHERHOUND", "HELLO WORLD", "A", "PARIS", "0 1 9"])
def test_encode_decode_round_trip(message):
    segments = encode_timing(message, dot_s=0.15)
    assert decode_timing(segments) == message


def test_encode_timing_matches_to_morse_symbol_counts():
    # "SOS" = "... --- ..." -> 9 marks (dots/dashes), 6 intra-symbol gaps,
    # 2 inter-character gaps; to_morse() gives us the reference pattern.
    assert to_morse("SOS") == "... --- ..."
    segments = encode_timing("SOS", dot_s=0.15)
    marks = [d for on, d in segments if on]
    assert len(marks) == 9


def test_decode_without_explicit_dot_s_estimates_from_shortest_mark():
    segments = encode_timing("SOS", dot_s=0.2)
    assert decode_timing(segments, dot_s=None) == "SOS"


def test_unknown_characters_are_dropped_like_to_morse():
    assert to_morse("SOS!") == to_morse("SOS")
    assert decode_timing(encode_timing("SOS!", dot_s=0.15)) == "SOS"


def test_empty_message_round_trips_to_empty_string():
    assert encode_timing("", dot_s=0.15) == []
    assert decode_timing([]) == ""


def test_presence_to_segments_collapses_runs():
    flags = [False, False, True, True, True, False, True, False]
    segments = presence_to_segments(flags, patch_duration_s=0.5)
    assert segments == [(False, 1.0), (True, 1.5), (False, 0.5), (True, 0.5), (False, 0.5)]


def test_decode_survives_patch_quantised_timing():
    """Simulate what the live pipeline actually sees: a per-patch (43 ms,
    matching dsp.patch_generator's 64-frame/32-hop/48kHz patches) boolean
    on/off sequence sampled from the continuous Morse envelope, rather than
    exact analog segment durations."""
    patch_dt = 64 * 32 / 48000  # ~43 ms, see dsp/patch_generator.py
    message = "SOS"
    segments = encode_timing(message, dot_s=0.15, lead_in_s=0.3)
    total = sum(d for _, d in segments)
    n_patches = int(total / patch_dt) + 2

    def is_on_at(time_s: float) -> bool:
        acc = 0.0
        for on, duration in segments:
            if time_s < acc + duration:
                return on
            acc += duration
        return False

    # Sample at each patch's center, like a real detector evaluating one
    # tone-presence decision per 43 ms patch.
    flags = [is_on_at((i + 0.5) * patch_dt) for i in range(n_patches)]

    decoded = decode_timing(presence_to_segments(flags, patch_dt))
    assert decoded == message


def test_decode_recovers_message_from_generated_attack_envelope():
    """End-to-end check against the exact tiling logic used by
    mock/generate_signals.py's infected=True branch (reproduced here since
    that module imports scipy at the top level and isn't needed for this
    timing-only check): build the same repeated on/off pulse_train and
    decode each repeated "SOS" back out."""
    sample_rate = 48000
    duration_sec = 20
    morse_dot_s = 0.15
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    pulse_train = np.zeros_like(t)
    message_segments = encode_timing("SOS", dot_s=morse_dot_s, lead_in_s=morse_dot_s * 2)
    repeat_gap_s = morse_dot_s * 14
    cursor = 0.0
    while cursor < duration_sec:
        for is_on, duration in message_segments:
            if cursor >= duration_sec:
                break
            start_idx = int(cursor * sample_rate)
            end_idx = min(int((cursor + duration) * sample_rate), len(t))
            if is_on and end_idx > start_idx:
                pulse_train[start_idx:end_idx] = 1.0
            cursor += duration
        cursor += repeat_gap_s

    patch_samples = 64 * 32
    n_patches = len(pulse_train) // patch_samples
    flags = [
        pulse_train[i * patch_samples:(i + 1) * patch_samples].mean() > 0.5
        for i in range(n_patches)
    ]
    patch_dt = patch_samples / sample_rate
    decoded = decode_timing(presence_to_segments(flags, patch_dt))

    # The 20 s clip fits ~3 full repeats; a final partial repeat at the tail
    # may decode to a stray short word, so check the repeated message is
    # there rather than requiring an exact whole-string match.
    words = decoded.split()
    assert words.count("SOS") >= 3
