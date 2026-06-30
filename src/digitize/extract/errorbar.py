"""Error-bar whisker extraction around known point locations.

The whisker is the stem (with caps) that passes *through* the marker. We isolate
it by following the connected run of mask pixels outward from the marker, inside a
narrow band centered on the marker. Because we trace only the run that touches the
marker, we don't pick up a neighbouring point's whisker, the connecting line, a
shaded band, or the axis — the failure modes of taking the whole strip's extent.

``axis`` selects vertical whiskers ('y' -> ``py_hi``/``py_lo``), horizontal ones
('x' -> ``xlo_px``/``xhi_px`` in ``extra``, e.g. a time confidence interval), or
'both'. Conversion to data space happens in the values step."""
from __future__ import annotations

import numpy as np


def extract_errorbars(search_mask: np.ndarray, points: list[dict], plot_box,
                      half_width: int | None = None, max_gap: int = 2,
                      axis: str = "y") -> list[dict]:
    x0, y0, bw, bh = [int(v) for v in plot_box]
    if half_width is None:
        half_width = max(2, int(0.003 * bw))
    half_height = max(2, int(0.003 * bh))
    out = []
    for p in points:
        px, py = int(round(p["px"])), int(round(p["py"]))
        rec = dict(p)
        if axis in ("y", "both"):
            pyr = py - y0
            lo_x, hi_x = max(x0, px - half_width), min(x0 + bw, px + half_width + 1)
            if 0 <= pyr < bh and hi_x > lo_x:
                col = search_mask[y0:y0 + bh, lo_x:hi_x].any(axis=1)
                start = _nearest_on(col, pyr)
                if start is not None:
                    rec["py_hi"] = float(_walk(col, start, -1, max_gap) + y0)
                    rec["py_lo"] = float(_walk(col, start, +1, max_gap) + y0)
        if axis in ("x", "both"):
            pxr = px - x0
            lo_y, hi_y = max(y0, py - half_height), min(y0 + bh, py + half_height + 1)
            if 0 <= pxr < bw and hi_y > lo_y:
                row = search_mask[lo_y:hi_y, x0:x0 + bw].any(axis=0)
                start = _nearest_on(row, pxr)
                if start is not None:
                    extra = dict(rec.get("extra") or {})
                    extra["xlo_px"] = float(_walk(row, start, -1, max_gap) + x0)
                    extra["xhi_px"] = float(_walk(row, start, +1, max_gap) + x0)
                    rec["extra"] = extra
        out.append(rec)
    return out


def _nearest_on(line: np.ndarray, idx: int) -> int | None:
    """``idx`` if on the mask, else the nearest on-mask index within 3 (handles an
    open marker whose centroid lands in the hollow), else None."""
    if line[idx]:
        return idx
    near = np.where(line[max(0, idx - 3):idx + 4])[0]
    if near.size == 0:
        return None
    return max(0, idx - 3) + int(near[near.size // 2])


def _walk(col: np.ndarray, start: int, step: int, max_gap: int) -> int:
    """Farthest index reachable from ``start`` along ``step`` while on the mask,
    tolerating up to ``max_gap`` consecutive off-mask pixels."""
    n = col.size
    last = start
    gap = 0
    r = start
    while 0 <= r + step < n:
        r += step
        if col[r]:
            last = r
            gap = 0
        else:
            gap += 1
            if gap > max_gap:
                break
    return last
