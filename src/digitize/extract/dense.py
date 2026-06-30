"""Techniques for dense, overlapping, and occluded multi-series figures.

These were distilled from real biomedical plots where curves cross, share error
bars, are drawn in near-identical shades, are sampled at non-uniform times, or are
partially hidden behind another series. Each is a deterministic helper the LLM can
compose with the colour/segmentation primitives:

- ``detect_markers``   markers at their true positions (handles non-uniform x).
- ``candidate_bands``  all curve/marker candidates in a column strip.
- ``track_series``     continuity tracking that rejects crossing-point outliers.
- ``fill_gaps``        slope-interpolate occluded points from the local trajectory.
- ``recover_circle``   locate an occluded circle's centre from its visible arc.
- ``trace_lines``      momentum tracking of monochrome overlapping lines.
- ``line_duty_cycle``  solid/dashed/dotted discrimination along a traced line.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max


def detect_markers(series_mask: np.ndarray, min_distance: int = 10,
                   min_radius: float = 2.3) -> list[dict]:
    """Marker centres as distance-transform peaks of the series mask.

    A filled marker is a *fat* blob (high distance to the nearest off-pixel); the
    connecting line and error-bar stems are thin (low distance). So peaks of the
    distance transform land on marker centres and ignore the lines between them.
    This recovers markers at their TRUE positions — essential when sampling is
    non-uniform (markers not on a regular x grid), where reading at assumed
    x-values would miss the peaks/troughs between them.
    """
    dist = ndimage.distance_transform_edt(series_mask.astype(bool))
    coords = peak_local_max(dist, min_distance=min_distance,
                            threshold_abs=min_radius, exclude_border=False)
    return [{"px": float(c[1]), "py": float(c[0]),
             "radius": float(dist[c[0], c[1]]), "source": "auto"} for c in coords]


def candidate_bands(strip: np.ndarray, frac: float = 0.25) -> list[tuple[float, float]]:
    """All contiguous row-bands above ``frac``*peak density in a vertical mask
    strip, as ``(row_center, total_pixels)``. These are the per-column candidates
    (one per curve passing through) that ``track_series`` chooses among."""
    rc = strip.sum(axis=1).astype(float)
    if rc.sum() == 0:
        return []
    thr = max(1.0, frac * rc.max())
    out, start = [], None
    for i, v in enumerate(rc):
        if v >= thr and start is None:
            start = i
        elif v < thr and start is not None:
            rows, w = np.arange(start, i), rc[start:i]
            out.append((float((rows * w).sum() / w.sum()), float(w.sum())))
            start = None
    if start is not None:
        rows, w = np.arange(start, rc.size), rc[start:]
        out.append((float((rows * w).sum() / w.sum()), float(w.sum())))
    return out


def _hampel(seq, k: float = 3.5, win: int = 3) -> np.ndarray:
    y = np.asarray(seq, float)
    flags = np.zeros(y.size, bool)
    for i in range(y.size):
        w = y[max(0, i - win):i + win + 1]
        med = np.median(w)
        mad = np.median(np.abs(w - med))
        if mad > 0 and abs(y[i] - med) > k * 1.4826 * mad:
            flags[i] = True
    return flags


def track_series(cands_per_col, xs, tol_px: float) -> list[float | None]:
    """Choose a smooth track through per-column candidate bands.

    The reference is a running median of the densest band per column (robust to a
    single-column spike even on a steep slope). A column is re-picked only when a
    *different* candidate sits genuinely closer to that trend — so real peaks
    (which have no alternative candidate) survive, while a crossing curve's marker
    (whose true band is a second candidate near the trend) is corrected. Returns a
    row position per column, or None where no plausible candidate exists (a gap to
    be filled by :func:`fill_gaps`)."""
    n = len(xs)
    dens = [max(c, key=lambda b: b[1])[0] if c else None for c in cands_per_col]
    idx = [i for i in range(n) if dens[i] is not None]
    if len(idx) < 3:
        return list(dens)
    di = np.interp(xs, [xs[i] for i in idx], [dens[i] for i in idx])
    ref = np.array([np.median(di[max(0, i - 1):i + 2]) for i in range(n)])
    extreme = {idx[j] for j, f in enumerate(_hampel([dens[i] for i in idx])) if f}
    out = []
    for i in range(n):
        c = cands_per_col[i]
        if not c:
            out.append(None)
            continue
        d = dens[i]
        near = min(c, key=lambda b: abs(b[0] - ref[i]))[0]
        if abs(d - ref[i]) > tol_px and abs(near - ref[i]) < abs(d - ref[i]) - 0.4 * tol_px:
            out.append(near)
        elif i in extreme and abs(d - ref[i]) > 2 * tol_px:
            out.append(None)
        else:
            out.append(d)
    return out


def fill_gaps(ys, xs):
    """Slope-interpolate None gaps from the local trajectory (linear between the
    nearest present points). Returns ``(filled, is_estimated)``; points outside
    the measured range stay None."""
    idx = [i for i in range(len(ys)) if ys[i] is not None]
    if len(idx) < 2:
        return list(ys), [False] * len(ys)
    xi, yi = [xs[i] for i in idx], [ys[i] for i in idx]
    out, est = [], []
    for i in range(len(ys)):
        if ys[i] is not None:
            out.append(ys[i]); est.append(False)
        elif xi[0] <= xs[i] <= xi[-1]:
            out.append(float(np.interp(xs[i], xi, yi))); est.append(True)
        else:
            out.append(None); est.append(False)
    return out, est


def _kasa(xs, ys):
    A = np.c_[2 * xs, 2 * ys, np.ones(len(xs))]
    a, b, c = np.linalg.lstsq(A, xs ** 2 + ys ** 2, rcond=None)[0]
    return a, b, float(np.sqrt(max(c + a * a + b * b, 0)))


def recover_circle(mask: np.ndarray, cx: float, cy_pred: float, radius: float,
                   win: int = 18) -> float | None:
    """Locate a partly-occluded filled-circle marker's centre row from its visible
    arc. Even a fraction of a circle of known ``radius`` constrains the full
    circle, so a least-squares (Kasa) fit of the arc pixels near a predicted centre
    returns the (hidden) centre; falls back to known-radius + visible-extent when
    the arc is too short to fit. Returns the centre row (py) or None."""
    x0, y0 = int(cx - win), int(cy_pred - win)
    sub = mask[max(0, y0):int(cy_pred + win), max(0, x0):int(cx + win)]
    ys, xs = np.where(sub)
    if len(xs) < 5:
        return None
    xs = xs.astype(float) + max(0, x0)
    ys = ys.astype(float) + max(0, y0)
    _, b, R = _kasa(xs, ys)
    if 0.6 * radius < R < 1.6 * radius and abs(b - cy_pred) < win:
        return float(b)
    span = ys.max() - ys.min()
    if span >= 2 * radius - 2:
        return float((ys.min() + ys.max()) / 2)
    return float(ys.min() + radius if cy_pred >= (ys.min() + ys.max()) / 2
                 else ys.max() - radius)


def trace_lines(mask: np.ndarray, x_seed: int, seed_rows, x_lo: int, x_hi: int,
                y_lo: int, y_hi: int, gate: float = 10.0, step: int = 1) -> dict:
    """Momentum tracking of monochrome OVERLAPPING lines (same colour, different
    line style). Seed one track per line where they're clearly separated, then
    extend each by the nearest mask pixel to a slope-extrapolated prediction — a
    smooth curve keeps its heading *through* a crossing rather than turning onto
    its neighbour. Returns ``{track_index: {x: row}}``."""
    tracks = {k: {x_seed: float(r)} for k, r in enumerate(seed_rows)}
    for sign in (-1, +1):
        pos = [float(r) for r in seed_rows]
        vel = [0.0] * len(seed_rows)
        x = x_seed
        while x_lo <= x + sign * step <= x_hi:
            x += sign * step
            rows = np.where(mask[y_lo:y_hi, max(0, x - 1):x + 2].any(axis=1))[0] + y_lo
            for k in range(len(pos)):
                pred = pos[k] + vel[k] * (sign * step)
                if rows.size:
                    nb = float(rows[np.argmin(np.abs(rows - pred))])
                    if abs(nb - pred) <= gate:
                        vel[k] = 0.6 * vel[k] + 0.4 * (nb - pos[k]) / (sign * step)
                        pos[k] = nb
                    else:
                        pos[k] = pred
                else:
                    pos[k] = pred
                pos[k] = min(max(pos[k], y_lo), y_hi)
                tracks[k][x] = pos[k]
    return tracks


def line_duty_cycle(mask: np.ndarray, track: dict, x_a: int, x_b: int) -> float:
    """Fraction of columns in [x_a, x_b] where the track's pixel is on the mask —
    discriminates line style: solid ≈ 1.0, dashed ≈ 0.5, dotted ≈ 0.3."""
    hit = tot = 0
    for x in range(x_a, x_b):
        if x not in track:
            continue
        py = int(round(track[x]))
        tot += 1
        if mask[max(0, py - 1):py + 2, x].any():
            hit += 1
    return hit / max(tot, 1)
