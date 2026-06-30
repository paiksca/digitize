"""Shared extraction helpers: mask cleanup, connected components, NMS."""
from __future__ import annotations

import cv2
import numpy as np


def clean_mask(mask: np.ndarray, open_k: int = 1, close_k: int = 2) -> np.ndarray:
    """Close small gaps then remove specks. Returns a boolean mask."""
    m = mask.astype(np.uint8)
    if close_k and close_k > 0:
        k = np.ones((close_k, close_k), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    if open_k and open_k > 0:
        k = np.ones((open_k, open_k), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    return m.astype(bool)


def components(mask: np.ndarray, min_area: int = 1) -> tuple[np.ndarray, list[dict]]:
    """Connected components with stats; filtered by ``min_area``."""
    n, labels, stats, cent = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        out.append({
            "label": i,
            "area": area,
            "cx": float(cent[i][0]),
            "cy": float(cent[i][1]),
            "bbox": [int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                     int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])],
        })
    return labels, out


def nms_points(cands: np.ndarray, min_dist: float) -> np.ndarray:
    """Greedy non-max suppression. ``cands`` rows are (x, y, score)."""
    if cands.shape[0] == 0:
        return cands
    order = np.argsort(-cands[:, 2])
    kept = []
    for i in order:
        x, y = cands[i, 0], cands[i, 1]
        if all((x - cands[j, 0]) ** 2 + (y - cands[j, 1]) ** 2 >= min_dist ** 2
               for j in kept):
            kept.append(i)
    return cands[kept]
