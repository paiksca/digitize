"""Kaplan-Meier (and other monotone step) curve extraction.

Traces the curve like any line, then enforces monotonicity — a survival curve
never increases, so the pixel row may only move one way along x. This cleans the
step trace. (Censoring-tick detection and Guyot IPD reconstruction, which also
need the numbers-at-risk table, are left to the operator for now.)"""
from __future__ import annotations

import numpy as np

from .line import trace_curve


def extract_km(series_mask, plot_box, resample: int | None = None,
               direction: str = "down") -> list[dict]:
    pts = trace_curve(series_mask, plot_box, resample=resample)
    if not pts:
        return pts
    pys = np.array([p["py"] for p in pts], float)
    # survival decreases over time -> data y falls -> pixel row only increases
    mono = np.maximum.accumulate(pys) if direction == "down" \
        else np.minimum.accumulate(pys)
    for p, m in zip(pts, mono):
        p["py"] = float(m)
    return pts
