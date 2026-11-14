"""Shared audio feature transformation for training, calibration, and inference."""
from __future__ import annotations

from pathlib import Path

import numpy as np


FEATURE_TRANSFORM = "log1p_ref_v1"


def load_reference_scale(path: str | Path = "models/ref_scale.txt") -> float:
    value = float(Path(path).read_text().strip())
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"reference scale must be a finite positive value: {path}")
    return value


def normalise_spectrogram(patches: np.ndarray, reference_scale: float) -> np.ndarray:
    """Map linear STFT magnitudes to the model feature distribution.

    The logarithm reduces domination by a few loud bins, while the fixed
    reference keeps samples comparable.  Never normalise a patch by itself:
    that would erase the very level and structure differences we detect.
    """
    if not np.isfinite(reference_scale) or reference_scale <= 0:
        raise ValueError("reference_scale must be finite and positive")
    magnitude = np.maximum(np.asarray(patches, dtype=np.float32), 0.0)
    return np.log1p(magnitude / reference_scale, dtype=np.float32) / np.float32(np.log(2.0))
