"""Image IO and geometry: loading, plot-box / axis-line detection, masks."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import config


def load_rgb(path: str | Path) -> np.ndarray:
    """Load an image as HxWx3 uint8 RGB (drops alpha onto white)."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        bgr = img[..., :3].astype(np.float32)
        alpha = img[..., 3:4].astype(np.float32) / 255.0
        img = (bgr * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def ink_mask(rgb: np.ndarray, thresh: int | None = None) -> np.ndarray:
    """Boolean mask of 'dark ink' pixels (axes, text, black markers/lines)."""
    gray = to_gray(rgb)
    if thresh is None:
        thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = min(int(thresh), 180)
    return gray < thresh


def nonwhite_mask(rgb: np.ndarray, white: int = 244) -> np.ndarray:
    """Pixels that are not near-white paper (i.e. any drawn content)."""
    return to_gray(rgb) < white


def content_bbox(rgb: np.ndarray) -> tuple[int, int, int, int]:
    """Bounding box (x, y, w, h) of all non-white content."""
    m = nonwhite_mask(rgb)
    ys, xs = np.where(m)
    if xs.size == 0:
        h, w = rgb.shape[:2]
        return 0, 0, w, h
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def detect_axis_lines(rgb: np.ndarray) -> tuple[list[int], list[int]]:
    """Return (vertical_line_cols, horizontal_line_rows) for long straight axes.

    Detected as columns/rows whose dark-ink count spans a large fraction of the
    image. Adjacent hits are merged to a single representative coordinate.
    """
    h, w = rgb.shape[:2]
    ink = ink_mask(rgb)
    col_counts = ink.sum(axis=0)
    row_counts = ink.sum(axis=1)
    vcols = _merge_runs(np.where(col_counts > config.AXIS_LINE_MIN_FRAC * h)[0])
    hrows = _merge_runs(np.where(row_counts > config.AXIS_LINE_MIN_FRAC * w)[0])
    return vcols, hrows


def _merge_runs(idx: np.ndarray, gap: int = 3) -> list[int]:
    """Collapse runs of nearby indices to their centers."""
    if idx.size == 0:
        return []
    idx = np.sort(idx)
    groups = []
    start = prev = idx[0]
    for v in idx[1:]:
        if v - prev <= gap:
            prev = v
        else:
            groups.append((start + prev) // 2)
            start = prev = v
    groups.append((start + prev) // 2)
    return [int(g) for g in groups]


def detect_plot_box(rgb: np.ndarray) -> dict:
    """Best-effort plot-area bounding box plus the raw detected axis lines.

    Heuristic: the plotting rectangle is bounded by the extreme long axis lines;
    where a side has no frame line we fall back to the content bounding box.
    This is only a *candidate* — the operator confirms it against the overview.
    """
    h, w = rgb.shape[:2]
    vcols, hrows = detect_axis_lines(rgb)
    cx0, cy0, cw, ch = content_bbox(rgb)
    cx1, cy1 = cx0 + cw, cy0 + ch

    x0 = vcols[0] if vcols else cx0
    x1 = vcols[-1] if len(vcols) >= 2 else cx1
    y0 = hrows[0] if len(hrows) >= 2 else cy0
    y1 = hrows[-1] if hrows else cy1
    x0, x1 = sorted((int(x0), int(x1)))
    y0, y1 = sorted((int(y0), int(y1)))
    box = [x0, y0, max(1, x1 - x0), max(1, y1 - y0)]
    return {
        "plot_box": box,
        "vertical_lines": vcols,
        "horizontal_lines": hrows,
        "content_bbox": [cx0, cy0, cw, ch],
        "image_size": [w, h],
    }


def _iou(a, b) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax0 + aw, bx0 + bw), min(ay0 + ah, by0 + bh)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _contained(a, b) -> float:
    """Fraction of box ``a`` that lies inside box ``b``."""
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax0 + aw, bx0 + bw), min(ay0 + ah, by0 + bh)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    return inter / max(1, aw * ah)


def detect_legend_candidates(rgb: np.ndarray, plot_box, max_n: int = 5) -> list[list[int]]:
    """Best-effort: framed, mostly-white rectangles inside the plot area.

    Legends typically sit inside the axes as a bordered box that is mostly paper
    with a little content (swatches + text). We return candidate boxes the
    operator can pass to ``extract --exclude`` after eyeballing the overlay; this
    is a *hint*, deliberately conservative, not an authority.
    """
    x0, y0, w, h = [int(v) for v in plot_box]
    a_plot = max(w * h, 1)
    gray = to_gray(rgb)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    white = ~nonwhite_mask(rgb)
    cands = []
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        bx, by, bw, bh = cv2.boundingRect(approx)
        area = bw * bh
        if area < 0.004 * a_plot or area > 0.45 * a_plot:
            continue
        if bx < x0 - 3 or by < y0 - 3 or bx + bw > x0 + w + 3 or by + bh > y0 + h + 3:
            continue
        rectness = cv2.contourArea(approx) / max(area, 1)
        if rectness < 0.85:
            continue
        white_frac = float(white[by:by + bh, bx:bx + bw].mean())
        if not (0.4 < white_frac < 0.97):  # mostly paper, but not empty
            continue
        cands.append(([int(bx), int(by), int(bw), int(bh)], area))
    cands.sort(key=lambda t: -t[1])
    out: list[list[int]] = []
    for box, _ in cands:
        if all(_iou(box, k) < 0.3 for k in out):
            out.append(box)
        if len(out) >= max_n:
            break
    return out


def _line_segments(mask: np.ndarray, axis: str, min_len: int) -> list[dict]:
    """Long straight segments along ``axis`` ('h' or 'v') via morphology."""
    if axis == "h":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_len))
    lines = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    n, _, stats, _ = cv2.connectedComponentsWithStats(lines, 8)
    segs = []
    for i in range(1, n):
        x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                      stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        if axis == "h" and w >= min_len:
            segs.append({"y": int(y + h // 2), "x0": int(x), "x1": int(x + w)})
        elif axis == "v" and h >= min_len:
            segs.append({"x": int(x + w // 2), "y0": int(y), "y1": int(y + h)})
    return segs


def detect_panels(rgb: np.ndarray, min_frac: float = 0.10, tol: int = 14) -> list[list[int]]:
    """Detect plot-AREA boxes for each panel of a (possibly multi-panel) figure.

    A panel is recognised by its corner: a bottom horizontal axis line whose left
    end meets a left vertical axis line. The box spans from that corner up/right to
    the lines' far ends. Works for full frames and bare L-shaped spines; returns
    boxes sorted top-to-bottom, left-to-right. Falls back to a single box from
    :func:`detect_plot_box` when no panel corner is found.
    """
    h, w = rgb.shape[:2]
    # medium threshold catches light-gray frames too (not just dark ink); the
    # long-kernel morphology rejects data points, which never form long lines
    mask = to_gray(rgb) < 200
    hsegs = _line_segments(mask, "h", max(20, int(min_frac * w)))
    vsegs = _line_segments(mask, "v", max(20, int(min_frac * h)))
    boxes = []
    for hs in hsegs:
        for vs in vsegs:
            # vertical line's x near the horizontal line's left end, and the
            # vertical spans up from (and reaches down to) the horizontal line
            if (abs(vs["x"] - hs["x0"]) <= tol and vs["y0"] < hs["y"] - tol
                    and vs["y1"] >= hs["y"] - tol):
                x0, y0 = vs["x"], vs["y0"]
                bw, bh = hs["x1"] - x0, hs["y"] - y0
                if bw > 0.12 * w and bh > 0.12 * h:  # reject slivers
                    boxes.append([int(x0), int(y0), int(bw), int(bh)])
    boxes.sort(key=lambda b: -b[2] * b[3])
    kept: list[list[int]] = []
    for b in boxes:
        # drop overlaps, shared-corner duplicates, and sub-rectangles mostly
        # contained in a larger panel (interior lines create false corners)
        if any(_iou(b, k) >= 0.4 for k in kept):
            continue
        if any(abs(b[0] - k[0]) <= tol and abs(b[1] - k[1]) <= tol for k in kept):
            continue
        if any(_contained(b, k) > 0.7 for k in kept):
            continue
        kept.append(b)
    if not kept:
        return [detect_plot_box(rgb)["plot_box"]]
    kept.sort(key=lambda b: (round(b[1] / max(1, 0.1 * h)), b[0]))  # row then column
    return kept


def box_mask(shape: tuple[int, int], box) -> np.ndarray:
    """Boolean mask True inside box (x, y, w, h)."""
    h, w = shape[:2]
    m = np.zeros((h, w), dtype=bool)
    x, y, bw, bh = [int(v) for v in box]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + bw), min(h, y + bh)
    m[y0:y1, x0:x1] = True
    return m


def subtract_boxes(mask: np.ndarray, boxes) -> np.ndarray:
    """Set mask False inside each exclusion box (e.g. an inset legend)."""
    out = mask.copy()
    for b in boxes or []:
        x, y, bw, bh = [int(v) for v in b]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(out.shape[1], x + bw), min(out.shape[0], y + bh)
        out[y0:y1, x0:x1] = False
    return out


def crop(rgb: np.ndarray, box) -> np.ndarray:
    x, y, bw, bh = [int(v) for v in box]
    h, w = rgb.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + bw), min(h, y + bh)
    return rgb[y0:y1, x0:x1].copy()
