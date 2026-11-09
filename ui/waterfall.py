"""Truecolour terminal waterfall. Newest row at the bottom."""
from __future__ import annotations

from collections import deque

import numpy as np
from rich.text import Text

from dsp.rf_pipeline import decimate_max

RAMP = [
    "#000004", "#0d0824", "#1f0c48", "#330a67", "#4a0c7a", "#611080",
    "#781d80", "#8c2c7c", "#a13d75", "#b44f6b", "#c56361", "#d47a58",
    "#df9350", "#e7ad4a", "#ecc947", "#f0e64e", "#f8f76a", "#fdff9a",
]


def _level(rgb_hex: str, ch: int) -> tuple[int, int, int]:
    h = rgb_hex.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


class Waterfall:
    def __init__(self, cfg):
        self.width = cfg.width
        self.height = cfg.height
        self.floor = cfg.floor_db
        self.ceil = cfg.ceil_db
        self._rows: deque[np.ndarray] = deque(maxlen=cfg.height)
        self._lut = [f"on {c}" for c in RAMP]

    def push(self, spectrum_db: np.ndarray) -> None:
        col = decimate_max(np.asarray(spectrum_db, dtype=np.float64), self.width)
        span = max(self.ceil - self.floor, 1e-9)
        norm = np.clip((col - self.floor) / span, 0.0, 1.0)
        idx = (norm * (len(self._lut) - 1)).round().astype(np.int8)
        self._rows.append(idx)

    def render(self, peak_col: int | None = None) -> Text:
        text = Text(no_wrap=True, overflow="crop")
        blank = " " * self.width

        for _ in range(self.height - len(self._rows)):
            text.append(blank + "\n")

        for row in self._rows:
            run_start = 0
            for i in range(1, self.width + 1):
                if i == self.width or row[i] != row[run_start]:
                    text.append(" " * (i - run_start), style=self._lut[int(row[run_start])])
                    run_start = i
            text.append("\n")

        if peak_col is not None and 0 <= peak_col < self.width:
            text.append(" " * peak_col + "^", style="bold white")
        return text