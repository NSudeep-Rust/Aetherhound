"""Unit tests for capture/rack_localization.py (virtual rack localization).

Pure numpy, no scipy/torch.
"""
from __future__ import annotations

import numpy as np
import pytest

from capture.rack_localization import (
    Position,
    RoomLayout,
    default_room,
    estimate_source_rack,
    expected_amplitude_ratios,
    simulate_mic_amplitudes,
)


def test_default_room_has_requested_counts():
    layout = default_room(rack_count=4, mic_count=3)
    assert len(layout.racks) == 4
    assert len(layout.mics) == 3
    assert layout.rack_names == ["rack-1", "rack-2", "rack-3", "rack-4"]


@pytest.mark.parametrize("rack_count,mic_count", [(2, 3), (4, 3), (4, 4), (6, 5)])
def test_noiseless_localization_is_always_correct(rack_count, mic_count):
    layout = default_room(rack_count=rack_count, mic_count=mic_count)
    for true_index in range(rack_count):
        observed = simulate_mic_amplitudes(layout, true_index, source_amplitude=1.7)
        guess_index, scores = estimate_source_rack(observed, layout)
        assert guess_index == true_index
        assert len(scores) == rack_count
        assert scores[layout.rack_names[true_index]] == max(scores.values())


def test_localization_is_invariant_to_source_loudness():
    """Different absolute source amplitudes at the same rack must resolve to
    the same rack -- only the ratio across mics should matter."""
    layout = default_room(rack_count=4, mic_count=3)
    for amplitude in (0.1, 1.0, 50.0):
        observed = simulate_mic_amplitudes(layout, 2, source_amplitude=amplitude)
        guess_index, _ = estimate_source_rack(observed, layout)
        assert guess_index == 2


def test_localization_degrades_gracefully_with_noise():
    layout = default_room(rack_count=4, mic_count=3)
    rng = np.random.default_rng(7)

    def accuracy(noise_floor: float, trials: int = 150) -> float:
        correct = 0
        for _ in range(trials):
            true_idx = rng.integers(0, len(layout.racks))
            observed = simulate_mic_amplitudes(
                layout, true_idx, source_amplitude=rng.uniform(0.5, 3.0),
                noise_floor=noise_floor, rng=rng,
            )
            guess, _ = estimate_source_rack(observed, layout)
            correct += guess == true_idx
        return correct / trials

    clean = accuracy(0.0)
    noisy = accuracy(0.1)
    assert clean == 1.0
    assert noisy < clean  # more mic noise must not improve accuracy


def test_expected_amplitude_ratios_sum_to_one():
    layout = default_room(rack_count=4, mic_count=3)
    for i in range(len(layout.racks)):
        ratios = expected_amplitude_ratios(layout, i)
        assert ratios.shape == (len(layout.mics),)
        assert ratios.min() >= 0
        assert abs(ratios.sum() - 1.0) < 1e-9


def test_closer_mic_sees_more_amplitude():
    # A second, far-off rack is included only to satisfy RoomLayout's
    # "at least 2 racks" invariant; the source is placed at rack 0.
    racks = (Position(0.0, 0.0), Position(100.0, 100.0))
    mics = (Position(1.0, 0.0), Position(10.0, 0.0), Position(0.5, 0.0))
    layout = RoomLayout(racks=racks, mics=mics)
    observed = simulate_mic_amplitudes(layout, 0, source_amplitude=1.0)
    # mic index 2 (distance 0.5) should see more amplitude than mic 0 (distance
    # 1.0), which in turn should see more than mic 1 (distance 10.0).
    assert observed[2] > observed[0] > observed[1]


def test_room_layout_rejects_too_few_racks_or_mics():
    with pytest.raises(ValueError):
        RoomLayout(racks=(Position(0, 0),), mics=(Position(0, 1), Position(1, 1), Position(2, 1)))
    with pytest.raises(ValueError):
        RoomLayout(racks=(Position(0, 0), Position(1, 0)), mics=(Position(0, 1), Position(1, 1)))


def test_estimate_source_rack_validates_input_shape_and_signal():
    layout = default_room(rack_count=4, mic_count=3)
    with pytest.raises(ValueError):
        estimate_source_rack(np.array([1.0, 2.0]), layout)  # wrong length
    with pytest.raises(ValueError):
        estimate_source_rack(np.zeros(3), layout)  # no signal
