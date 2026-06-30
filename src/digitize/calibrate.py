"""Axis calibration: detect tick/gridline candidates, snap operator guesses to
them, and build the pixel<->data transform with a reprojection-error report.

The LLM reads each tick's *value* (it is good at that) and gives an approximate
pixel location; the snapper moves that location onto the nearest detected tick so
the transform rests on sub-pixel-accurate anchors (the LLM is bad at that).
"""
from __future__ import annotations

import numpy as np

from . import config
from .imaging import _merge_runs, ink_mask, nonwhite_mask
from .transform import AxisTransform


def detect_tick_sources(rgb: np.ndarray, plot_box, axis: str) -> tuple[list[int], list[int]]:
    """Return ``(marks, grid)`` candidate positions along ``axis``.

    ``marks`` are dark tick marks just outside the spine, length-filtered to the
    *major* (longer) ticks — these correspond to the labelled values. ``grid``
    are full-length interior gridlines, which on log/semilog plots include minor
    lines and so are a less trustworthy snap target.
    """
    x0, y0, bw, bh = [int(v) for v in plot_box]
    x1, y1 = x0 + bw, y0 + bh
    ink = ink_mask(rgb)
    nw = nonwhite_mask(rgb)
    h, w = rgb.shape[:2]
    marks: list[int] = []
    grid: list[int] = []

    if axis == "x":
        lo, hi = max(0, y1 + 1), min(h, y1 + 1 + config.TICK_STRIP)
        if hi > lo:
            cc = ink[lo:hi, x0:x1].sum(axis=0)
            thr = max(2, int(0.5 * cc.max())) if cc.size and cc.max() else 2
            marks = [int(c + x0) for c in np.where(cc >= thr)[0]]
        interior = nw[y0:y1, x0:x1]
        if interior.shape[0] > 0:
            cc2 = interior.sum(axis=0)
            grid = [int(c + x0) for c in np.where(cc2 > 0.6 * bh)[0]]
    elif axis == "y":
        lo, hi = max(0, x0 - config.TICK_STRIP), max(0, x0 - 1)
        if hi > lo:
            rc = ink[y0:y1, lo:hi].sum(axis=1)
            thr = max(2, int(0.5 * rc.max())) if rc.size and rc.max() else 2
            marks = [int(r + y0) for r in np.where(rc >= thr)[0]]
        interior = nw[y0:y1, x0:x1]
        if interior.shape[1] > 0:
            rc2 = interior.sum(axis=1)
            grid = [int(r + y0) for r in np.where(rc2 > 0.6 * bw)[0]]
    else:
        raise ValueError("axis must be 'x' or 'y'")

    return (_merge_runs(np.array(sorted(set(marks))), gap=3),
            _merge_runs(np.array(sorted(set(grid))), gap=3))


def detect_ticks(rgb: np.ndarray, plot_box, axis: str) -> list[int]:
    """All candidate tick positions (marks + grid), merged."""
    marks, grid = detect_tick_sources(rgb, plot_box, axis)
    return _merge_runs(np.array(sorted(set(marks) | set(grid))), gap=3)


def snap(coord: float, marks: list[int], grid: list[int] | None = None,
         radius: int = config.TICK_SNAP_RADIUS) -> tuple[float, bool]:
    """Snap ``coord`` to the nearest tick mark within ``radius``; fall back to a
    gridline only if no mark is close (gridlines may be minor on log axes)."""
    for cands in (marks, grid or []):
        if cands:
            arr = np.asarray(cands, float)
            j = int(np.argmin(np.abs(arr - coord)))
            if abs(arr[j] - coord) <= radius:
                return float(arr[j]), True
    return float(coord), False


def build_calibration(rgb, plot_box, x_refs, y_refs, scale_x="linear",
                      scale_y="linear", do_snap=True, mode="separable",
                      affine_points=None) -> dict:
    """Build a calibration.

    ``x_refs``/``y_refs``: lists of {'px'/'py': float, 'val': float}.
    ``affine_points`` (affine mode): list of {'px','py','x','y'}.
    Returns a dict with the serialized transform and per-ref snap info.
    """
    if mode == "affine":
        pts = [(p["px"], p["py"], p["x"], p["y"]) for p in (affine_points or [])]
        t = AxisTransform.fit_affine(pts, scale_x, scale_y)
        return {"transform": t.to_dict(), "mode": "affine",
                "affine_points": affine_points,
                "reprojection_rms_px": t.reprojection_rms_px}

    xm, xg = detect_tick_sources(rgb, plot_box, "x") if do_snap else ([], [])
    ym, yg = detect_tick_sources(rgb, plot_box, "y") if do_snap else ([], [])
    x_used, y_used = [], []
    x_pairs, y_pairs = [], []
    for r in x_refs:
        px = float(r["px"])
        spx, snapped = (snap(px, xm, xg) if do_snap else (px, False))
        x_used.append({"px_input": px, "px": spx, "val": float(r["val"]),
                       "snapped": snapped})
        x_pairs.append((spx, float(r["val"])))
    for r in y_refs:
        py = float(r["py"])
        spy, snapped = (snap(py, ym, yg) if do_snap else (py, False))
        y_used.append({"py_input": py, "py": spy, "val": float(r["val"]),
                       "snapped": snapped})
        y_pairs.append((spy, float(r["val"])))

    t = AxisTransform.fit_separable(x_pairs, y_pairs, scale_x, scale_y)
    return {
        "transform": t.to_dict(),
        "mode": "separable",
        "x_refs": x_used,
        "y_refs": y_used,
        "tick_candidates": {"x_marks": xm, "x_grid": xg, "y_marks": ym, "y_grid": yg},
        "reprojection_rms_px": t.reprojection_rms_px,
    }
