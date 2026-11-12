"""Truecolour wideband waterfall. Invalid (DC-guard) bins render dim."""
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
GAP = "on #23272f"


class Waterfall:
    def __init__(self, cfg):
        self.width = cfg.width
        self.height = cfg.height
        self.floor = cfg.floor_db
        self.ceil = cfg.ceil_db
        self._rows: deque[np.ndarray] = deque(maxlen=cfg.height)
        self._lut = [f"on {c}" for c in RAMP]
        # Fixed dB range for stable rendering (no adaptive flicker / stripes)
        self._floor_db = cfg.floor_db
        self._ceil_db = cfg.ceil_db

    def _style(self, level: int) -> str:
        return GAP if level < 0 else self._lut[level]

    def push(self, frame) -> None:
        db = decimate_max(np.asarray(frame.db, dtype=np.float64), self.width)
        v = decimate_max(frame.valid.astype(np.float64), self.width) > 0.5
        
        # Fixed dB range mapping (no adaptive range = no vertical stripes)
        span = max(self._ceil_db - self._floor_db, 1e-9)
        col = np.full(self.width, -200.0)
        ok = np.zeros(self.width, dtype=bool)
        col[v] = db[v]
        ok[v] = True

        norm = np.clip((col - self._floor_db) / span, 0.0, 1.0)
        idx = (norm * (len(self._lut) - 1)).round().astype(np.int8)
        idx[~ok] = -1
        self._rows.append(idx)

    def render(self, peak_col: int | None = None, labels: list | None = None) -> Text:
        text = Text(no_wrap=True, overflow="crop")
        for _ in range(self.height - len(self._rows)):
            text.append(" " * self.width + "\n")

        for row in self._rows:
            start = 0
            for i in range(1, self.width + 1):
                if i == self.width or row[i] != row[start]:
                    text.append(" " * (i - start), style=self._style(int(row[start])))
                    start = i
            text.append("\n")

        # Always emit caret line (fixed line count prevents vertical shake)
        caret = [" "] * self.width
        if peak_col is not None and 0 <= peak_col < self.width:
            caret[peak_col] = "^"
        text.append("".join(caret) + "\n", style="bold white")
        if labels:
            # Place labels under their respective bands
            text.append(" | ".join(labels), style="dim")
        return text