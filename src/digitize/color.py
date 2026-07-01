"""Color-based series separation — the robustness core for messy biomed figures.

Biomedical panels routinely pack many series in close colors with idiosyncratic
legends (color patches, line samples, marker samples, inside/outside the axes).
We handle that with three operator-facing primitives:

* ``sample_swatch``  — robustly read the color under a point the LLM indicates.
* ``detect_legend_swatches`` — auto-find colored marks in a legend region and
  return their colors + locations, so the LLM can label them.
* ``dominant_palette`` — k-means the chromatic pixels in the plot area to propose
  series colors when the legend is unusable.

Segmentation uses a *nearest-target* rule in CIE Lab: each series color competes
with background anchors (white paper, and — only when no series is itself dark —
black and mid-gray for axes/text/gridlines). A pixel joins a series only if that
series is its nearest target AND within tolerance. This stops a chromatic series
from vacuuming up antialiased gray gridlines, which naive thresholding does.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import config
from .imaging import crop
from .util import lab_chroma, rgb_to_hex, rgb_to_lab


@dataclass
class SeriesColorModel:
    name: str
    rgb: np.ndarray  # (3,) float
    lab: np.ndarray  # (3,) float

    @property
    def hex(self) -> str:
        return rgb_to_hex(self.rgb)

    @property
    def chroma(self) -> float:
        return float(lab_chroma(self.lab))

    @classmethod
    def from_rgb(cls, name: str, rgb) -> "SeriesColorModel":
        rgb = np.asarray(rgb, dtype=float)
        return cls(name=name, rgb=rgb, lab=rgb_to_lab(rgb).astype(float))


def sample_swatch(rgb: np.ndarray, px: float, py: float, radius: int = 4) -> dict:
    """Read the dominant non-white color in a small patch around (px, py)."""
    x, y = int(round(px)), int(round(py))
    patch = crop(rgb, [x - radius, y - radius, 2 * radius + 1, 2 * radius + 1])
    if patch.size == 0:
        raise ValueError(f"swatch patch at ({px},{py}) is empty")
    lab = rgb_to_lab(patch)
    chroma = lab_chroma(lab).reshape(-1)
    L = lab[..., 0].reshape(-1)
    flat = patch.reshape(-1, 3).astype(float)
    keep = ~((L > config.WHITE_L) & (chroma < config.LOW_CHROMA))
    sel = flat[keep] if keep.any() else flat
    color = np.median(sel, axis=0)
    return {"color": rgb_to_hex(color), "rgb": color.tolist(),
            "chroma": float(lab_chroma(rgb_to_lab(color)))}


def detect_legend_swatches(rgb: np.ndarray, box, min_area: int = 6,
                           merge_tol: float = 10.0) -> list[dict]:
    """Find chromatic marks (color/line/marker samples) in a legend region.

    Returns one entry per distinct color: ``{color, px, py, area, count}`` in
    full-image pixel coordinates, sorted by total area. Black/grey text is
    rejected by the chroma filter, so this targets the common colored legend.
    """
    x, y, bw, bh = [int(v) for v in box]
    sub = crop(rgb, box)
    lab = rgb_to_lab(sub)
    chroma = lab_chroma(lab)
    mask = (chroma > config.LOW_CHROMA).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blobs = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp = labels == i
        med_rgb = np.median(sub[comp].astype(float), axis=0)
        cx, cy = centroids[i]
        blobs.append({"rgb": med_rgb, "px": float(cx + x), "py": float(cy + y),
                      "area": area})
    # merge blobs of (nearly) the same color — a series' marker + line sample
    merged: list[dict] = []
    for b in sorted(blobs, key=lambda d: -d["area"]):
        blab = rgb_to_lab(b["rgb"])
        for m in merged:
            if np.linalg.norm(blab - rgb_to_lab(m["rgb"])) < merge_tol:
                tot = m["area"] + b["area"]
                m["px"] = (m["px"] * m["area"] + b["px"] * b["area"]) / tot
                m["py"] = (m["py"] * m["area"] + b["py"] * b["area"]) / tot
                m["rgb"] = (np.asarray(m["rgb"]) * m["area"]
                            + np.asarray(b["rgb"]) * b["area"]) / tot
                m["area"] = tot
                m["count"] += 1
                break
        else:
            merged.append({**b, "count": 1})
    return [{"color": rgb_to_hex(m["rgb"]), "px": round(m["px"], 1),
             "py": round(m["py"], 1), "area": m["area"], "count": m["count"]}
            for m in sorted(merged, key=lambda d: -d["area"])]


def dominant_palette(rgb: np.ndarray, mask: np.ndarray, k: int = 6) -> list[dict]:
    """k-means the chromatic pixels inside ``mask`` to propose series colors."""
    lab = rgb_to_lab(rgb)
    chroma = lab_chroma(lab)
    sel = mask & (chroma > config.LOW_CHROMA)
    pts = rgb[sel].astype(np.float32)
    if pts.shape[0] < k:
        return []
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    # sample for speed on large images
    if pts.shape[0] > 50000:
        idx = np.linspace(0, pts.shape[0] - 1, 50000).astype(int)
        pts_fit = pts[idx]
    else:
        pts_fit = pts
    _, labels, centers = cv2.kmeans(pts_fit, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.ravel(), minlength=k)
    frac = counts / counts.sum()
    order = np.argsort(-frac)
    return [{"color": rgb_to_hex(centers[i]), "rgb": centers[i].tolist(),
             "fraction": round(float(frac[i]), 4)} for i in order]


def segment_series(rgb: np.ndarray, plot_mask: np.ndarray,
                   models: list[SeriesColorModel], tol: float | None = None,
                   include_dark_bg: bool = True) -> dict[str, np.ndarray]:
    """Assign plot-area pixels to series via nearest-target in Lab.

    Returns ``{series_name: bool_mask}``. Background anchors (white, plus black &
    mid-gray when no series is dark) absorb paper/axes/text/grid pixels.
    """
    if tol is None:
        tol = config.DEFAULT_COLOR_TOL
    lab = rgb_to_lab(rgb)
    flat = lab.reshape(-1, 3)
    series_labs = [m.lab for m in models]
    any_dark = any(m.chroma < config.LOW_CHROMA * 1.5 and m.lab[0] < 35
                   for m in models)
    bg = [np.array([100.0, 0.0, 0.0])]  # white paper
    if include_dark_bg and not any_dark:
        bg += [np.array([0.0, 0.0, 0.0]),    # black axes/text
               np.array([60.0, 0.0, 0.0])]   # mid-grey gridlines
    targets = series_labs + bg
    dists = np.stack([np.linalg.norm(flat - t, axis=1) for t in targets], axis=1)
    nearest = np.argmin(dists, axis=1)
    dmin = dists[np.arange(flat.shape[0]), nearest]
    pm = plot_mask.reshape(-1)
    out: dict[str, np.ndarray] = {}
    for i, m in enumerate(models):
        sel = (nearest == i) & (dmin < tol) & pm
        out[m.name] = sel.reshape(lab.shape[:2])
    return out
