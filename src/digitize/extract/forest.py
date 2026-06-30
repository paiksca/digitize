"""Forest-plot extraction: one horizontal row per study/subgroup, each a point
estimate (a marker) sitting on a confidence-interval line.

Per row we read the CI extent (leftmost..rightmost colored pixel) and the point
estimate (the column block with the greatest vertical extent — the marker is
taller than the thin CI line). Reported in pixel space; the values step maps the
three x's to estimate / CI-low / CI-high."""
from __future__ import annotations

import numpy as np

from .boxplot import _max_run
from .common import clean_mask


def extract_forest(series_mask: np.ndarray, plot_box, min_ci_frac: float = 0.015,
                   row_gap: int = 4) -> list[dict]:
    x0, y0, bw, bh = [int(v) for v in plot_box]
    mask = clean_mask(series_mask, open_k=0, close_k=2)
    rowext = np.array([_max_run(mask[y, x0:x0 + bw]) for y in range(y0, y0 + bh)])
    present = rowext > max(4, min_ci_frac * bw)
    rows = np.where(present)[0]
    if rows.size == 0:
        return []
    # cluster present rows into forest rows
    bands, start, prev = [], rows[0], rows[0]
    for r in rows[1:]:
        if r - prev <= row_gap:
            prev = r
        else:
            bands.append((start, prev))
            start = prev = r
    bands.append((start, prev))

    out = []
    for ry0, ry1 in bands:
        sub = mask[y0 + ry0:y0 + ry1 + 1, x0:x0 + bw]
        present = sub.any(axis=0)
        if not present.any():
            continue
        # the CI is the longest CONTIGUOUS run of columns; isolated pixels from a
        # vertical spine crossing the row are shorter runs and ignored
        idx = np.where(present)[0]
        splits = np.where(np.diff(idx) > 1)[0]
        segs = np.split(idx, splits + 1)
        ci = max(segs, key=len)
        lo, hi = int(ci[0]), int(ci[-1])
        if hi - lo >= 0.9 * bw:  # a full-width line is an axis spine, not a CI
            continue
        colsum = sub.sum(axis=0)
        colsum[:lo] = 0
        colsum[hi + 1:] = 0  # restrict the marker search to the CI line
        mcols = np.where(colsum >= max(2, 0.6 * colsum.max()))[0]
        point = float(mcols.mean()) if mcols.size else (lo + hi) / 2.0
        out.append({"py": float(y0 + (ry0 + ry1) / 2.0),
                    "point_px": float(x0 + point), "lo_px": float(x0 + lo),
                    "hi_px": float(x0 + hi), "source": "auto"})
    out.sort(key=lambda r: r["py"])
    return out
