"""Numeric quality report to accompany the visual round-trip overlay."""
from __future__ import annotations

import numpy as np

from .schemas import Manifest, SeriesData


def series_stats(sd: SeriesData) -> dict:
    xs = np.array([p.x for p in sd.points if p.x is not None], float)
    ys = np.array([p.y for p in sd.points if p.y is not None], float)
    if xs.size == 0:
        return {"name": sd.name, "kind": sd.kind, "n": 0}
    stat = {
        "name": sd.name, "kind": sd.kind, "n": int(xs.size),
        "x_range": [float(xs.min()), float(xs.max())],
        "y_range": [float(ys.min()), float(ys.max())],
        "x_monotonic": bool(np.all(np.diff(xs[np.argsort(xs)]) >= 0)),
    }
    rels = [p.y_err / abs(p.y) for p in sd.points
            if p.y_err and p.y not in (None, 0)]
    if rels:
        stat["median_rel_y_uncertainty"] = round(float(np.median(rels)), 4)
        stat["max_rel_y_uncertainty"] = round(float(np.max(rels)), 4)
    return stat


def report(manifest: Manifest, calib: dict, series_list: list[SeriesData]) -> dict:
    rms = calib.get("reprojection_rms_px")
    flags = []
    if rms is not None and rms > 2.0:
        flags.append(f"high calibration reprojection error ({rms:.2f}px) — "
                     "recheck tick values / snapping")
    stats = [series_stats(s) for s in series_list]
    for s in stats:
        if s["n"] == 0:
            flags.append(f"series '{s['name']}' has no calibrated points")
        elif s["kind"] != "bar" and not s.get("x_monotonic", True) and s["kind"] == "line":
            flags.append(f"series '{s['name']}' x not monotonic — possible "
                         "trace error or double-back")
    return {"reprojection_rms_px": rms, "series": stats, "flags": flags}
