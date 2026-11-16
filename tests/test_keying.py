"""Unit tests for dsp/keying.py's on/off toggling detector.

No numpy/scipy/torch needed here -- KeyingDetector is plain Python -- so
these run in any environment that can import pytest at all.
"""
from __future__ import annotations

import pytest

from dsp.keying import KeyingDetector


def test_steady_tone_never_flags():
    # Fan hum / a missed mains harmonic: always present, same frequency.
    kd = KeyingDetector(window_patches=50, min_bursts=2, freq_tolerance_hz=100)
    assert not any(kd.update(True, 4000.0)[0] for _ in range(50))


def test_isolated_blip_never_flags():
    kd = KeyingDetector(window_patches=50, min_bursts=2, freq_tolerance_hz=100)
    sequence = [False] * 10 + [True] * 3 + [False] * 20
    flags = [kd.update(present, 1875.0 if present else 0.0)[0] for present in sequence]
    assert not any(flags)


def test_genuine_keying_flags_with_correct_frequency():
    kd = KeyingDetector(window_patches=50, min_bursts=2, freq_tolerance_hz=100)
    sequence = [False] * 5 + [True] * 5 + [False] * 5 + [True] * 5 + [False] * 5
    is_keyed, freq = False, None
    for present in sequence:
        is_keyed, freq = kd.update(present, 4031.25 if present else 0.0)
    assert is_keyed
    assert freq == pytest.approx(4031.25)


def test_unrelated_frequency_blips_do_not_falsely_cluster():
    kd = KeyingDetector(window_patches=50, min_bursts=2, freq_tolerance_hz=50)
    sequence = [(False, 0.0), (True, 1000.0), (False, 0.0), (True, 9000.0), (False, 0.0)]
    is_keyed = False
    for present, hz in sequence:
        is_keyed, _ = kd.update(present, hz)
    assert not is_keyed


def test_flips_true_exactly_when_second_burst_closes():
    kd = KeyingDetector(window_patches=50, min_bursts=2, freq_tolerance_hz=50)
    sequence = [False, True, False, True, False, True, False]
    flags = [kd.update(present, 4031.25 if present else 0.0)[0] for present in sequence]
    assert flags == [False, False, False, True, True, True, True]


def test_reset_clears_history():
    kd = KeyingDetector(window_patches=50, min_bursts=2, freq_tolerance_hz=50)
    for present in (False, True, False, True):
        kd.update(present, 4031.25 if present else 0.0)
    kd.reset()
    is_keyed, freq = kd.update(True, 4031.25)
    assert not is_keyed
    assert freq is None


def test_rejects_invalid_construction():
    with pytest.raises(ValueError):
        KeyingDetector(window_patches=1)
    with pytest.raises(ValueError):
        KeyingDetector(min_bursts=1)
    with pytest.raises(ValueError):
        KeyingDetector(freq_tolerance_hz=-1)
