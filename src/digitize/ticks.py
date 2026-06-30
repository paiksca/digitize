"""Automatic tick-LABEL localization.

The slow part of calibrating a real figure is finding, to the pixel, where each
axis tick label sits. Reading the *value* off a label is trivial for a vision
model; locating its center precisely is not. So this module does the latter and
leaves the former to the operator:

* find the band of text just outside an axis (below it for x, left of it for y),
* cluster that band into individual labels,
* return each label's center coordinate along the axis.

The operator then reads the values off the rendered overlay and passes them with
``digitize ticks --axis x --values ...``; the tool zips them with the detected
positions to build calibration references. No OCR, no by-eye pixel reading.
"""
from __future__ import annotations

import cv2
import numpy as np

from .util import rgb_to_lab


def _first_run(present: np.ndarray, allow_gap: int = 2):
    """First contiguous run of True (tolerating small gaps). Returns (start,end)."""
    idx = np.where(present)[0]
    if idx.size == 0:
        return None
    start = end = idx[0]
    for v in idx[1:]:
        if v - end <= allow_gap + 1:
            end = v
        else:
            break
    return int(start), int(end)


def _cluster(indices: np.ndarray, gap: int):
    if indices.size == 0:
        return []
    groups = [[int(indices[0])]]
    for v in indices[1:]:
        if v - groups[-1][-1] <= gap:
            groups[-1].append(int(v))
        else:
            groups.append([int(v)])
    return groups


def detect_axis_labels(rgb: np.ndarray, plot_box, axis: str, max_strip: int = 90,
                       dark_thresh: float = 60.0, side: str = "left",
                       chroma_thresh: float = 18.0) -> dict:
    """Locate tick-label centers along ``axis`` ('x' or 'y').

    Returns ``{"labels": [{"pos": float, "bbox": [x,y,w,h]}], "band": [x,y,w,h]}``
    with labels sorted along the axis (left->right for x, top->bottom for y).
    """
    H, W = rgb.shape[:2]
    x0, y0, bw, bh = [int(v) for v in plot_box]
    x1, y1 = x0 + bw, y0 + bh
    # "text" = dark OR chromatic, so colored axis labels (e.g. a blue right-hand
    # efficacy axis) are detected, not just black ones.
    lab = rgb_to_lab(rgb)
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    dark = (lab[..., 0] < dark_thresh) | (chroma > chroma_thresh)

    if axis == "x":
        top = min(H - 1, y1 + 2)
        strip = dark[top:min(H, top + max_strip), x0:x1]
        # the tick-label line is the FIRST text row below the axis; the axis title
        # is a separate row further down (allow_gap=1 keeps them apart). Exclude
        # near-full-width rows so a solid axis line isn't mistaken for labels.
        rsum = strip.sum(axis=1)
        # threshold skips sparse tick MARKS (a few px/row); the <0.7*bw cap rejects
        # a solid axis line, so _first_run lands on the label line (above the title)
        present = (rsum > max(3, 0.02 * bw)) & (rsum < 0.7 * bw)
        run = _first_run(present, allow_gap=1)
        if run is None:
            return {"labels": [], "band": None}
        r0, r1 = run
        thick = max(4, r1 - r0 + 1)
        band_top = top + r0
        sub = dark[band_top:band_top + thick, x0:x1].astype(np.uint8)
        # close inter-character gaps so each number is one run; labels stay apart
        k = max(3, int(0.6 * thick))
        closed = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, np.ones((1, k), np.uint8))
        cols = np.where(closed.sum(axis=0) > 0)[0]
        groups = _cluster(cols, gap=2)
        labels = []
        for g in groups:
            if len(g) < max(2, int(0.2 * thick)):
                continue
            labels.append({"pos": float((g[0] + g[-1]) / 2 + x0),
                           "bbox": [int(x0 + g[0]), int(band_top),
                                    int(g[-1] - g[0] + 1), int(thick)]})
        labels.sort(key=lambda d: d["pos"])
        return {"labels": labels, "band": [x0, int(band_top), bw, int(thick)]}

    # y-axis: in the strip beside the spine, keep moderate-fill columns (drops the
    # rotated title and the spine), take the column-run nearest the spine (the
    # tick labels), and cluster its rows into one label per tick. ``side`` selects
    # the left axis (default) or a right-hand axis (e.g. a dual-axis plot).
    if side == "right":
        left = min(W - 1, x1 + 1)
        right = min(W, x1 + max_strip)
    else:
        right = max(1, x0 - 1)
        left = max(0, x0 - max_strip)
    sub = dark[y0:y1, left:right]
    colsum = sub.sum(axis=0)
    keep = (colsum > max(2, 0.02 * bh)) & (colsum < 0.5 * bh)
    if not keep.any():
        return {"labels": [], "band": None}
    ki = np.where(keep)[0]
    runs = _cluster(ki, gap=max(4, int(0.02 * (right - left))))
    substantial = [r for r in runs if len(r) >= 3]
    # labels are the run nearest the spine: rightmost for a left axis, leftmost
    # for a right axis.
    band_cols = (substantial or runs)[0 if side == "right" else -1]
    bc0, bc1 = band_cols[0], band_cols[-1]
    sub = dark[y0:y1, left + bc0:left + bc1 + 1]
    left = left + bc0
    rows = np.where(sub.sum(axis=1) > 0)[0]
    groups = _cluster(rows, gap=max(4, int(0.012 * H)))
    band_left = int(left)
    band_w = int(bc1 - bc0 + 1)
    labels = []
    for g in groups:
        if len(g) < 2:
            continue
        labels.append({"pos": float((g[0] + g[-1]) / 2 + y0),
                       "bbox": [band_left, int(y0 + g[0]), band_w,
                                int(g[-1] - g[0] + 1)]})
    labels.sort(key=lambda d: d["pos"])
    return {"labels": labels, "band": [band_left, y0, band_w, bh]}
