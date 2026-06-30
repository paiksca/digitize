"""Discrete-marker extraction: blob detection with merged-marker splitting, and
template matching for shape-coded / monochrome series."""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from .. import config
from .common import clean_mask, components, nms_points


def _classify_shape(submask: np.ndarray) -> tuple[str, float]:
    """Classify a marker's outline as circle/triangle/diamond/square (works for
    filled or open markers — the external contour carries the shape). Returns the
    name and circularity 4*pi*A/P^2."""
    sub = submask.astype(np.uint8)
    cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return "other", 0.0
    c = max(cnts, key=cv2.contourArea)
    A, P = cv2.contourArea(c), cv2.arcLength(c, True)
    if A < 4 or P <= 0:
        return "other", 0.0
    circ = float(4 * np.pi * A / (P * P))
    nv = len(cv2.approxPolyDP(c, 0.05 * P, True))
    # a filled square sits at circ~0.78; require a higher bar for "circle" so
    # squares fall through to the 4-vertex branch instead of being mislabelled.
    if circ >= 0.82:
        return "circle", circ
    if nv <= 3:
        return "triangle", circ
    if nv == 4:
        return "diamond", circ
    return ("circle" if circ > 0.6 else "other"), circ


def extract_markers(series_mask: np.ndarray, min_area: int | None = None,
                    max_area: int | None = None, split_merged: bool = True,
                    split_factor: float | None = None, open_k: int = 1,
                    shape: str | None = None) -> list[dict]:
    """Centroids of discrete markers in a per-series boolean mask.

    Blobs much larger than ``split_factor`` x the median area (overlapping
    markers) are split with a distance-transform watershed so dense clusters
    don't collapse to one point. Lower ``split_factor`` for tightly-spaced
    markers; raise it if single fat markers are being over-split.

    ``open_k`` sets the morphological-opening kernel: raise it (3-5) to erase a
    fit line or error-bar whiskers of the same color that would otherwise connect
    markers into one blob — the fatter marker cores survive.

    ``shape`` (circle/triangle/diamond/square) keeps only markers of that outline
    — use it to split two same-color series drawn with different shapes (e.g.
    filled circles vs open triangles), and it incidentally drops non-circular
    error-bar caps when set to ``circle``. Each returned point carries its
    detected ``shape``.
    """
    min_area = config.MIN_MARKER_AREA if min_area is None else min_area
    split_factor = config.MERGE_SPLIT_FACTOR if split_factor is None else split_factor
    mask = clean_mask(series_mask, open_k=open_k, close_k=2)
    _, comps = components(mask, min_area=min_area)
    if not comps:
        return []
    med = float(np.median([c["area"] for c in comps]))
    pts: list[dict] = []
    for c in comps:
        if max_area and c["area"] > max_area and not split_merged:
            continue
        x, y, w, h = c["bbox"]
        shp, _circ = _classify_shape(mask[y:y + h, x:x + w])
        big = split_merged and med > 0 and c["area"] > med * split_factor
        if big:
            pts.extend({**q, "shape": shp} for q in _split_blob(mask, c, med))
        else:
            pts.append({"px": c["cx"], "py": c["cy"], "area": c["area"],
                        "shape": shp, "source": "auto"})
    if shape:
        pts = [p for p in pts if p.get("shape") == shape]
    return pts


def _split_blob(mask: np.ndarray, comp: dict, med_area: float) -> list[dict]:
    x, y, w, h = comp["bbox"]
    sub = mask[y:y + h, x:x + w].astype(np.uint8)
    dist = cv2.distanceTransform(sub, cv2.DIST_L2, 3)
    k = max(1, int(round(comp["area"] / max(med_area, 1.0))))
    min_d = max(2, int(np.sqrt(med_area / np.pi)))
    coords = peak_local_max(dist, num_peaks=k, min_distance=min_d, labels=sub > 0)
    if len(coords) <= 1:
        return [{"px": comp["cx"], "py": comp["cy"], "area": comp["area"],
                 "source": "auto"}]
    markers = np.zeros(dist.shape, dtype=np.int32)
    for idx, (ry, rx) in enumerate(coords, 1):
        markers[ry, rx] = idx
    ws = watershed(-dist, markers, mask=sub > 0)
    out = []
    for idx in range(1, int(markers.max()) + 1):
        m = ws == idx
        a = int(m.sum())
        if a < config.MIN_MARKER_AREA:
            continue
        ys, xs = np.where(m)
        out.append({"px": float(xs.mean() + x), "py": float(ys.mean() + y),
                    "area": a, "source": "auto"})
    return out


def match_markers(rgb: np.ndarray, template: np.ndarray, threshold: float = 0.6,
                  roi_mask: np.ndarray | None = None) -> list[dict]:
    """Template-match a sample marker crop; NMS the peaks to marker centers.

    Use for monochrome figures where series differ by *shape* not color: the LLM
    points at one exemplar marker, the tool finds the rest.
    """
    th, tw = template.shape[:2]
    res = cv2.matchTemplate(rgb, template, cv2.TM_CCOEFF_NORMED)
    ys, xs = np.where(res >= threshold)
    if xs.size == 0:
        return []
    cx = xs + tw / 2.0
    cy = ys + th / 2.0
    if roi_mask is not None:
        ok = roi_mask[np.clip(cy.astype(int), 0, roi_mask.shape[0] - 1),
                      np.clip(cx.astype(int), 0, roi_mask.shape[1] - 1)]
        cx, cy, sc = cx[ok], cy[ok], res[ys, xs][ok]
    else:
        sc = res[ys, xs]
    cands = np.column_stack([cx, cy, sc])
    kept = nms_points(cands, min_dist=min(th, tw) * 0.6)
    return [{"px": float(c[0]), "py": float(c[1]), "score": float(c[2]),
             "source": "auto"} for c in kept]
