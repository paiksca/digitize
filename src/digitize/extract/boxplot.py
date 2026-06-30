"""Box-and-whisker extraction.

Each box is a connected component; within it the box body edges (Q1, Q3) and the
median are the *full-width* horizontal lines, while the whisker stem/caps are
narrow. We read those levels per box and report them in pixel space (the values
step converts them). Works for vertical or horizontal boxes (the latter is the
vertical analysis on a transposed mask)."""
from __future__ import annotations

import numpy as np

from .common import clean_mask, components


def _max_run(row: np.ndarray) -> int:
    """Length of the longest True run in a boolean row."""
    if not row.any():
        return 0
    idx = np.where(row)[0]
    splits = np.where(np.diff(idx) > 1)[0]
    runs = np.split(idx, splits + 1)
    return max(len(r) for r in runs)


def _runs(full: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous (start, end) True runs in a boolean column."""
    idx = np.where(full)[0]
    if idx.size == 0:
        return []
    out, start, prev = [], idx[0], idx[0]
    for v in idx[1:]:
        if v - prev <= 2:
            prev = v
        else:
            out.append((int(start), int(prev)))
            start = prev = v
    out.append((int(start), int(prev)))
    return out


def _box_levels(mask: np.ndarray, bbox) -> dict | None:
    x, y, w, h = bbox
    sub = mask[y:y + h, x:x + w]
    extents = np.array([_max_run(sub[r]) for r in range(h)], float)
    if extents.max() <= 0:
        return None
    box_w = extents.max()
    full = _runs(extents >= 0.55 * box_w)  # box edges/median (filled body or lines)
    if not full:
        return None
    tall = [r for r in full if r[1] - r[0] >= 4]  # filled body half/halves
    median = None
    if tall:  # FILLED box: body spans the tall band(s); caps are thin (excluded)
        q3, q1 = tall[0][0], tall[-1][1]
        if len(tall) >= 2:  # a white median splits the fill -> gap between halves
            median = (tall[0][1] + tall[1][0]) / 2.0
        else:
            inner = extents[q3 + 2:q1 - 1]
            if inner.size and inner.min() < 0.7 * box_w:
                median = q3 + 2 + int(np.argmin(inner))
    else:  # OUTLINED box: thin full-width lines are the edges + median
        centers = [(a + b) / 2.0 for a, b in full]
        q3, q1 = centers[0], centers[-1]
        if len(centers) >= 3:
            median = centers[len(centers) // 2]
    return {"center": x + w / 2.0, "q3": float(q3 + y), "q1": float(q1 + y),
            "median": float(median + y) if median is not None else None,
            "whis_hi": float(y), "whis_lo": float(y + h - 1)}


def extract_boxes(series_mask: np.ndarray, plot_box, orient: str = "v",
                  min_area: int = 30) -> list[dict]:
    """One entry per box. Vertical: q1/q3/median/whisker as pixel ROWS (py) and
    center as px. Horizontal: the same but as pixel COLUMNS (px) and center py."""
    mask = clean_mask(series_mask, open_k=0, close_k=2)
    work = mask.T if orient == "h" else mask
    _, comps = components(work, min_area=min_area)
    boxes = []
    for c in comps:
        lv = _box_levels(work, c["bbox"])
        if lv is None:
            continue
        if orient == "h":
            boxes.append({"py": lv["center"], "q1_px": lv["q1"], "q3_px": lv["q3"],
                          "median_px": lv["median"], "whis_lo_px": lv["whis_hi"],
                          "whis_hi_px": lv["whis_lo"], "source": "auto"})
        else:
            boxes.append({"px": lv["center"], "q1_py": lv["q1"], "q3_py": lv["q3"],
                          "median_py": lv["median"], "whis_lo_py": lv["whis_lo"],
                          "whis_hi_py": lv["whis_hi"], "source": "auto"})
    boxes.sort(key=lambda b: b.get("px", b.get("py")))
    return boxes
