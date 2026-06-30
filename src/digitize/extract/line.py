"""Continuous-curve tracing from a per-series mask.

For each pixel column we take the *longest contiguous run* of series pixels and
use its midpoint. Taking the longest run (rather than the mean of all rows)
keeps the trace on the line through thick strokes and avoids averaging across a
gap where the curve doubles back or another feature intrudes.
"""
from __future__ import annotations

import numpy as np

from .common import clean_mask


def _hampel(y: np.ndarray, window: int = 7, k: float = 4.0) -> np.ndarray:
    """Replace isolated outliers (stray same-color pixels in a column) with the
    local median. Conservative (k=4 MAD) so real multi-column peaks survive."""
    y = np.asarray(y, float)
    if y.size < window:
        return y
    out = y.copy()
    half = window // 2
    for i in range(y.size):
        lo, hi = max(0, i - half), min(y.size, i + half + 1)
        w = y[lo:hi]
        med = np.median(w)
        mad = np.median(np.abs(w - med))
        if mad > 0 and abs(y[i] - med) > k * 1.4826 * mad:
            out[i] = med
    return out


def _runs(ys: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous (start, end) runs in a sorted array of row indices."""
    runs = []
    start = prev = ys[0]
    for v in ys[1:]:
        if v == prev + 1:
            prev = v
        else:
            runs.append((start, prev))
            start = prev = v
    runs.append((start, prev))
    return runs


def trace_curve(series_mask: np.ndarray, plot_box, resample: int | None = None,
                at_px: list[float] | None = None, smooth: int = 0,
                edge: str = "center", declutter: bool = True) -> list[dict]:
    """Trace a curve column-by-column within the plot box.

    ``resample`` returns N evenly x-spaced points; ``at_px`` samples at given
    pixel columns (used when the operator asks for specific data x-values);
    ``smooth`` applies a moving average of that window over the raw trace.

    ``edge`` selects what each column contributes: ``"center"`` (midpoint of the
    longest run — best for a drawn line), ``"top"`` (uppermost series pixel) or
    ``"bottom"`` (lowermost). Use ``"top"`` to trace the upper envelope of a
    shaded/filled region (e.g. a model band) rather than its midline.
    """
    x0, y0, bw, bh = [int(v) for v in plot_box]
    x1, y1 = x0 + bw, y0 + bh
    mask = clean_mask(series_mask, open_k=0, close_k=2)
    cols, centers, tops, bots = [], [], [], []
    for px in range(x0, x1):
        ys = np.where(mask[y0:y1, px])[0]
        if ys.size == 0:
            continue
        s, e = max(_runs(ys), key=lambda r: r[1] - r[0])  # longest run
        cols.append(px)
        centers.append((s + e) / 2.0 + y0)
        tops.append(float(ys.min()) + y0)   # uppermost series pixel (envelope)
        bots.append(float(ys.max()) + y0)    # lowermost
    if not cols:
        return []
    cols = np.asarray(cols, float)
    primary = {"top": tops, "bottom": bots}.get(edge, centers)
    series = {"py": np.asarray(primary, float)}
    if edge == "band":
        series["py_hi"] = np.asarray(tops, float)
        series["py_lo"] = np.asarray(bots, float)

    if declutter:  # kill isolated stray-pixel spikes before smoothing/resampling
        for k in series:
            series[k] = _hampel(series[k])
    if smooth and smooth > 1 and cols.size >= smooth:
        from scipy.ndimage import uniform_filter1d
        # mode="nearest" avoids the zero-padding edge artifact that np.convolve
        # ("same") introduces, which would drag the endpoints toward pixel 0.
        for k in series:
            series[k] = uniform_filter1d(series[k], size=int(smooth), mode="nearest")

    if at_px is not None:
        xs_new = np.asarray(at_px, float)
    elif resample:
        xs_new = np.linspace(cols.min(), cols.max(), int(resample))
    else:
        xs_new = cols
    interp = {k: np.interp(xs_new, cols, v) for k, v in series.items()}
    out = []
    for i, c in enumerate(xs_new):
        pt = {"px": float(c), "py": float(interp["py"][i]), "source": "auto"}
        if edge == "band":
            pt["py_hi"] = float(interp["py_hi"][i])
            pt["py_lo"] = float(interp["py_lo"][i])
        out.append(pt)
    return out
