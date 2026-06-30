"""Heatmap extraction: a grid of color-coded cells whose values are read from a
colorbar (color scale), not from x/y axes.

The operator gives the grid region + rows/cols and the colorbar region + value
range. Cells are sampled by their *dominant* (modal) color, so thin inter-cell
gridlines don't dilute them, and cell boundaries may be uniform, explicit
(``col_edges``/``row_edges``), or auto-detected from color changes (``auto_grid``)
so heatmaps with unequal cell widths work. Each cell maps to the value of its
nearest colorbar color in CIE Lab. Returns the value matrix."""
from __future__ import annotations

import numpy as np

from ..imaging import crop
from ..util import rgb_to_lab


def _cell_color(rgb, x0, y0, x1, y1):
    """Dominant (modal) color of a cell — robust to thin gridlines/specks."""
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    sub = rgb[max(0, y0):max(y0 + 1, y1), max(0, x0):max(x0 + 1, x1)]
    if sub.size == 0:
        return np.array([255.0, 255.0, 255.0])
    flat = sub.reshape(-1, 3).astype(int)
    q = flat // 16  # 16 levels/channel
    keys = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]
    vals, counts = np.unique(keys, return_counts=True)
    mode = vals[int(counts.argmax())]
    return flat[keys == mode].astype(float).mean(axis=0)


def _count_cells(rgb, box, axis, max_n=90):
    """Estimate the cell count along ``axis`` from the gridline spacing.

    Heatmap cells are separated by a border color, so a cell boundary is a line
    that is *uniform along the perpendicular direction* (a vertical gridline is
    one color top-to-bottom; a cell-interior column crosses cells of differing
    values and isn't). Those lines show as valleys in the per-line variance
    profile; the median valley spacing is the cell size, and L / spacing the
    count — robust even when a few valleys are missed or spurious."""
    x, y, w, h = [int(v) for v in box]
    sub = crop(rgb, box).astype(float)
    var = (sub.var(axis=0) if axis == "col" else sub.var(axis=1)).mean(axis=1)
    L = w if axis == "col" else h
    rng = float(var.max() - var.min())
    if L < 8 or rng < 1e-6:
        return 1
    from scipy.signal import find_peaks
    valleys, _ = find_peaks(-var, distance=max(2, L // max_n),
                            prominence=0.12 * rng)
    if valleys.size < 2:
        return 1
    period = float(np.median(np.diff(valleys)))
    return max(1, int(round(L / period))) if period >= 2 else 1


def _edges(rgb, box, n, axis):
    """``n+1`` cell boundaries along ``axis`` ('col'/'row') from the strongest
    color-change positions (handles unequal cell sizes)."""
    x, y, w, h = [int(v) for v in box]
    sub = crop(rgb, box).astype(float)
    prof = sub.mean(axis=0) if axis == "col" else sub.mean(axis=1)
    L = w if axis == "col" else h
    change = np.concatenate([[0.0], np.linalg.norm(np.diff(prof, axis=0), axis=1)])
    min_sp = max(2, int(0.5 * L / max(n, 1)))
    chosen = []
    for idx in np.argsort(-change):
        if len(chosen) >= n - 1:
            break
        if idx < min_sp or idx > L - min_sp:
            continue
        if all(abs(int(idx) - c) >= min_sp for c in chosen):
            chosen.append(int(idx))
    base = x if axis == "col" else y
    return [float(base)] + [float(base + c) for c in sorted(chosen)] + [float(base + L)]


def sample_colorbar(rgb, box, vmin, vmax, orient="v", reverse=False, n=64, log=False):
    """Return [(value, lab_color)] sampled along the colorbar."""
    x, y, w, h = [int(v) for v in box]
    ramp = []
    for i in range(n):
        f = i / (n - 1)
        if orient == "v":
            py = y + h - 1 - int(f * (h - 1))
            px = x + w // 2
        else:
            px = x + int(f * (w - 1))
            py = y + h // 2
        g = f if not reverse else 1 - f
        value = vmin * (vmax / vmin) ** g if log else vmin + g * (vmax - vmin)
        ramp.append((value, rgb_to_lab(rgb[py, px].astype(float))))
    return ramp


def extract_heatmap(rgb, grid_box, n_rows, n_cols, colorbar_box, vmin, vmax,
                    cbar_orient="v", cbar_reverse=False, log=False,
                    col_edges=None, row_edges=None, auto_grid=False) -> dict:
    if vmin == vmax:
        raise ValueError("vmin and vmax must differ")
    if log and (vmin <= 0 or vmax <= 0):
        raise ValueError("log color scale needs positive vmin and vmax")
    x, y, w, h = [float(v) for v in grid_box]
    if n_cols <= 0:  # auto-detect cell count from the repeat period
        n_cols = _count_cells(rgb, grid_box, "col")
    if n_rows <= 0:
        n_rows = _count_cells(rgb, grid_box, "row")
    if col_edges is None:
        col_edges = (_edges(rgb, grid_box, n_cols, "col") if auto_grid
                     else [x + i * w / n_cols for i in range(n_cols + 1)])
    if row_edges is None:
        row_edges = (_edges(rgb, grid_box, n_rows, "row") if auto_grid
                     else [y + i * h / n_rows for i in range(n_rows + 1)])
    ramp = sample_colorbar(rgb, colorbar_box, vmin, vmax, cbar_orient,
                           cbar_reverse, log=log)
    ramp_lab = np.array([c for _, c in ramp])
    ramp_val = np.array([v for v, _ in ramp])
    matrix, centers = [], []
    for r in range(n_rows):
        row_vals, row_cen = [], []
        for c in range(n_cols):
            cx0, cx1 = col_edges[c], col_edges[c + 1]
            cy0, cy1 = row_edges[r], row_edges[r + 1]
            mx, my = (cx1 - cx0) * 0.2, (cy1 - cy0) * 0.2  # central 60% of the cell
            color = _cell_color(rgb, cx0 + mx, cy0 + my, cx1 - mx, cy1 - my)
            lab = rgb_to_lab(color)
            best = int(np.argmin(np.linalg.norm(ramp_lab - lab, axis=1)))
            row_vals.append(float(ramp_val[best]))
            row_cen.append([float((cx0 + cx1) / 2), float((cy0 + cy1) / 2)])
        matrix.append(row_vals)
        centers.append(row_cen)
    return {"matrix": matrix, "centers": centers, "n_rows": n_rows,
            "n_cols": n_cols, "col_edges": col_edges, "row_edges": row_edges}
