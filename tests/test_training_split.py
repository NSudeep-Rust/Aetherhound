"""Unit tests for the group-independent held-out split (train/run_training.py).

Importing train.run_training pulls in torch (requirements-train.txt only,
not the base requirements-dev.txt test dependencies), so skip this whole
module instead of failing collection when torch isn't installed.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from train.run_training import split_indices_by_group


def test_split_indices_by_group_keeps_whole_recordings_out():
    # 5 recordings, 20 overlapping patches each -- like the 50%-overlap
    # sliding window mock/generate_signals.py slices each synthetic clip with.
    groups = np.repeat(np.arange(5), 20)
    train_idx, val_idx = split_indices_by_group(groups, validation_fraction=0.2, seed=7)

    train_groups = set(groups[train_idx].tolist())
    val_groups = set(groups[val_idx].tolist())
    assert train_groups.isdisjoint(val_groups), "a recording's patches leaked across the split"
    assert len(train_idx) > 0 and len(val_idx) > 0
    assert len(train_idx) + len(val_idx) == len(groups)


def test_split_indices_by_group_is_reproducible():
    groups = np.repeat(np.arange(8), 15)
    first = split_indices_by_group(groups, validation_fraction=0.25, seed=7)
    second = split_indices_by_group(groups, validation_fraction=0.25, seed=7)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_different_split_seeds_can_choose_different_recordings():
    # The split seed is deliberately separate from the model's
    # torch.manual_seed -- changing one must not silently move the other.
    groups = np.repeat(np.arange(6), 10)
    val_a = set(groups[split_indices_by_group(groups, 0.2, seed=7)[1]].tolist())
    val_b = set(groups[split_indices_by_group(groups, 0.2, seed=99)[1]].tolist())
    assert val_a != val_b


def test_split_indices_by_group_requires_multiple_recordings():
    groups = np.zeros(10, dtype=np.int32)
    with pytest.raises(ValueError):
        split_indices_by_group(groups, validation_fraction=0.2, seed=7)
