"""Render annotated overlays. These PNGs are the channel through which the LLM
*verifies* the tool: high-contrast, labelled, and sized for vision inspection."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from . import config  # noqa: E402
from .imaging import crop, to_gray  # noqa: E402
from .transform import AxisTransform  # noqa: E402

# distinct, high-contrast colors for series whose color is unknown / for markup
PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46f0f0",
           "#f032e6", "#bcf60c", "#fabebe", "#008080", "#9a6324", "#800000"]


def _fig(rgb, dim=False, figscale=7.0):
    h, w = rgb.shape[:2]
    fig, ax = plt.subplots(figsize=(figscale, figscale * h / max(w, 1)))
    if dim:
        ax.imshow(to_gray(rgb), cmap="gray", vmin=0, vmax=255, alpha=0.55)
        ax.set_facecolor("white")
    else:
        ax.imshow(rgb)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xlabel("pixel x")
    ax.set_ylabel("pixel y")
    return fig, ax


def _save(fig, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=config.OVERLAY_DPI)
    plt.close(fig)
    return str(out_path)


def _nice_values(lo, hi, scale, n=6):
    lo, hi = sorted((float(lo), float(hi)))
    if scale == "log10" and lo > 0:
        k0, k1 = math.floor(math.log10(lo)), math.ceil(math.log10(hi))
        return [10.0**k for k in range(k0, k1 + 1) if lo <= 10.0**k <= hi]
    loc = MaxNLocator(nbins=n, steps=[1, 2, 2.5, 5, 10])
    return [v for v in loc.tick_values(lo, hi) if lo <= v <= hi]


def _box_rect(ax, box, color, label=None, ls="-"):
    x, y, w, h = box
    ax.add_patch(mpatches.Rectangle((x, y), w, h, fill=False, edgecolor=color,
                                    lw=2, ls=ls))
    if label:
        ax.text(x + 2, y - 4, label, color=color, fontsize=9, weight="bold")


# --- overview --------------------------------------------------------------
def render_overview(rgb, info, out_path, legends=None):
    fig, ax = _fig(rgb)
    _box_rect(ax, info["plot_box"], "#ffcc00", "plot_box (candidate)")
    for c in info.get("vertical_lines", []):
        ax.axvline(c, color="#00d0ff", lw=0.8, alpha=0.7)
    for r in info.get("horizontal_lines", []):
        ax.axhline(r, color="#ff5bd0", lw=0.8, alpha=0.7)
    for b in legends or []:
        _box_rect(ax, b, "#ff7f0e", "legend?", ls="--")
    ax.set_title("plot box (yellow) + long axis lines; orange = possible legend "
                 "to --exclude. Override --plot-box if wrong.", fontsize=9)
    return _save(fig, out_path)


# --- calibration -----------------------------------------------------------
def render_calibration(rgb, plot_box, calib, out_path):
    t = AxisTransform.from_dict(calib["transform"])
    fig, ax = _fig(rgb)
    _box_rect(ax, plot_box, "#ffcc00", "plot_box")
    x0, y0, w, h = [int(v) for v in plot_box]
    x1, y1 = x0 + w, y0 + h
    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2

    # reprojected gridlines at nice data values
    xa_lo, _ = t.pixel_to_data(x0, ym)
    xa_hi, _ = t.pixel_to_data(x1, ym)
    _, ya_top = t.pixel_to_data(xm, y0)
    _, ya_bot = t.pixel_to_data(xm, y1)
    for vx in _nice_values(xa_lo, xa_hi, t.scale_x):
        px, _ = t.data_to_pixel(vx, ya_bot)
        ax.axvline(float(px), color="#00e0a0", lw=0.7, alpha=0.8)
        ax.text(float(px), y1 + 12, _fmt(vx), color="#00a070", fontsize=7,
                ha="center", rotation=90)
    for vy in _nice_values(ya_bot, ya_top, t.scale_y):
        _, py = t.data_to_pixel(xa_lo, vy)
        ax.axhline(float(py), color="#00e0a0", lw=0.7, alpha=0.8)
        ax.text(x0 - 8, float(py), _fmt(vy), color="#00a070", fontsize=7,
                ha="right", va="center")

    # reference ticks: input (open) vs snapped (filled)
    for r in calib.get("x_refs", []):
        ax.plot([r["px_input"]], [y1], "o", mfc="none", mec="#ff3030", ms=9)
        ax.plot([r["px"]], [y1], "o", color="#ff3030", ms=5)
        ax.text(r["px"], y1 + 2, f' {_fmt(r["val"])}', color="#ff3030", fontsize=7)
    for r in calib.get("y_refs", []):
        ax.plot([x0], [r["py_input"]], "o", mfc="none", mec="#ff3030", ms=9)
        ax.plot([x0], [r["py"]], "o", color="#ff3030", ms=5)
        ax.text(x0 + 4, r["py"], f' {_fmt(r["val"])}', color="#ff3030", fontsize=7)

    rms = calib.get("reprojection_rms_px", 0.0)
    ax.set_title(f"Calibration — reprojection RMS = {rms:.2f}px. Green grid should "
                 f"sit on the figure's gridlines; red = ticks used.", fontsize=9)
    return _save(fig, out_path)


# --- extraction ------------------------------------------------------------
def render_extraction(rgb, plot_box, series_render, out_path, exclude=None):
    fig, ax = _fig(rgb, dim=True)
    _box_rect(ax, plot_box, "#999900", "plot_box")
    for b in exclude or []:
        _box_rect(ax, b, "#ff0000", "excluded", ls="--")
    handles = []
    for i, s in enumerate(series_render):
        color = s.get("color") or PALETTE[i % len(PALETTE)]
        pts = s["points"]
        if not pts:
            continue
        xs = [p["px"] for p in pts]
        ys = [p["py"] for p in pts]
        if s.get("kind") == "line":
            ax.plot(xs, ys, "-", color=color, lw=1.6, alpha=0.9)
            ax.plot(xs[:: max(1, len(xs) // 40)], ys[:: max(1, len(xs) // 40)],
                    ".", color=color, ms=3)
        else:
            ax.scatter(xs, ys, s=42, facecolors="none", edgecolors=color,
                       linewidths=1.8)
            step = 1 if len(pts) <= 60 else 5
            for j, p in enumerate(pts):
                if j % step == 0:
                    ax.text(p["px"] + 4, p["py"] - 4, str(j), color=color,
                            fontsize=6.5)
        handles.append(mpatches.Patch(color=color,
                       label=f'{s["name"]} (n={len(pts)})'))
    if handles:
        ax.legend(handles=handles, fontsize=8, loc="best", framealpha=0.85)
    ax.set_title("Extracted points (open=marker, numbered for edit). Check for "
                 "misses / false positives on gridlines.", fontsize=9)
    return _save(fig, out_path)


# --- verify (round-trip) ---------------------------------------------------
def render_verify(rgb, plot_box, manifest, series_list, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    sub = crop(rgb, plot_box)
    axes[0].imshow(sub)
    axes[0].set_title("Original (plot area)", fontsize=10)
    axes[0].set_xticks([]); axes[0].set_yticks([])

    ax = axes[1]
    for i, sd in enumerate(series_list):
        color = sd.color or PALETTE[i % len(PALETTE)]
        xs = [p.x for p in sd.points if p.x is not None]
        ys = [p.y for p in sd.points if p.y is not None]
        if not xs:
            continue
        if sd.kind == "line":
            order = np.argsort(xs)
            ax.plot(np.array(xs)[order], np.array(ys)[order], "-", color=color,
                    label=sd.name)
        else:
            yerr = [p.y_err if p.y_err else 0 for p in sd.points if p.x is not None]
            if any(yerr):
                ax.errorbar(xs, ys, yerr=yerr, fmt="o", color=color, ms=4,
                            capsize=2, label=sd.name)
            else:
                ax.plot(xs, ys, "o", color=color, ms=4, label=sd.name)
    if manifest.x.scale == "log10":
        ax.set_xscale("log")
    if manifest.y.scale == "log10":
        ax.set_yscale("log")
    ax.set_xlabel(f"{manifest.x.name} [{manifest.x.unit}]")
    ax.set_ylabel(f"{manifest.y.name} [{manifest.y.unit}]")
    ax.set_title("Reconstructed from extracted data", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    if series_list:
        ax.legend(fontsize=8)
    fig.suptitle("Round-trip check — the two panels should look like the same plot",
                 fontsize=11)
    return _save(fig, out_path)


# --- fit -------------------------------------------------------------------
def render_fit(series_data, fit, manifest, out_path):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    color = series_data.color or PALETTE[0]
    xs = [p.x for p in series_data.points if p.x is not None]
    ys = [p.y for p in series_data.points if p.y is not None]
    yerr = [p.y_err if p.y_err else 0 for p in series_data.points if p.x is not None]
    ax.errorbar(xs, ys, yerr=yerr if any(yerr) else None, fmt="o", color=color,
                ms=5, capsize=2, label="data", zorder=3)
    if "curve" in fit:
        ax.plot(fit["curve"]["x"], fit["curve"]["y"], "-", color="#222", lw=2,
                label=f'{fit["model"]} fit')
    if manifest.x.scale == "log10":
        ax.set_xscale("log")
    if manifest.y.scale == "log10":
        ax.set_yscale("log")
    ax.set_xlabel(f"{manifest.x.name} [{manifest.x.unit}]")
    ax.set_ylabel(f"{manifest.y.name} [{manifest.y.unit}]")
    bits = []
    for k, v in fit.get("derived", {}).items():
        if isinstance(v, (int, float)):
            bits.append(f"{k}={_fmt(v)}")
    r2 = fit.get("r2")
    title = f'{series_data.name}: {fit["model"]}'
    if r2 is not None:
        title += f"  (R²={r2:.3f})"
    ax.set_title(title + ("\n" + ", ".join(bits) if bits else ""), fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    return _save(fig, out_path)


# --- palette / swatches ----------------------------------------------------
def render_palette(colors, out_path, title="Detected colors"):
    n = len(colors)
    fig, ax = plt.subplots(figsize=(6, max(1.5, 0.5 * n + 0.5)))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, max(1, n))
    ax.axis("off")
    for i, c in enumerate(colors):
        y = n - 1 - i
        ax.add_patch(mpatches.Rectangle((0.2, y + 0.1), 1.4, 0.8, color=c["color"]))
        extra = []
        if "px" in c:
            extra.append(f'@({c["px"]:.0f},{c["py"]:.0f})')
        if "fraction" in c:
            extra.append(f'{c["fraction"]*100:.0f}%')
        if "area" in c:
            extra.append(f'area={c["area"]}')
        ax.text(1.8, y + 0.5, f'{c["color"]}  ' + "  ".join(extra),
                va="center", fontsize=10, family="monospace")
    ax.set_title(title, fontsize=11)
    return _save(fig, out_path)


# --- zoom ------------------------------------------------------------------
def save_zoom(rgb, box, scale, out_path):
    import cv2

    sub = crop(rgb, box)
    if sub.size == 0:
        raise ValueError("zoom region is empty")
    big = cv2.resize(sub, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    fig, ax = plt.subplots(figsize=(8, 8 * big.shape[0] / max(big.shape[1], 1)))
    ax.imshow(big)
    ax.set_title(f"Zoom x{scale} of region {list(box)} — read tick values here",
                 fontsize=9)
    # pixel ruler in ORIGINAL coordinates
    x, y, w, h = [int(v) for v in box]
    ax.set_xticks(np.linspace(0, big.shape[1], 6))
    ax.set_xticklabels([f"{int(x + v / scale)}" for v in np.linspace(0, big.shape[1], 6)])
    ax.set_yticks(np.linspace(0, big.shape[0], 6))
    ax.set_yticklabels([f"{int(y + v / scale)}" for v in np.linspace(0, big.shape[0], 6)])
    return _save(fig, out_path)


def render_grid(rgb, out_path, step=200):
    """The image overlaid with a labelled pixel grid, so the operator can read
    sub-panel boxes off dense composites that `panels`/`auto` under-detect."""
    h, w = rgb.shape[:2]
    fig, ax = plt.subplots(figsize=(min(18, 16), min(18, 16 * h / max(w, 1))))
    ax.imshow(rgb)
    for x in range(0, w, step):
        ax.axvline(x, color="#39ff14", lw=0.5, alpha=0.6)
        ax.text(x, max(8, step * 0.05), str(x), color="red", fontsize=6, rotation=90)
    for y in range(0, h, step):
        ax.axhline(y, color="#39ff14", lw=0.5, alpha=0.6)
        ax.text(2, y, str(y), color="red", fontsize=6, va="bottom")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"pixel grid every {step}px — read sub-panel boxes (x,y,w,h)",
                 fontsize=9)
    return _save(fig, out_path)


def render_panels(rgb, panels, out_path):
    """Numbered panel plot-area boxes over the figure, for verification."""
    fig, ax = _fig(rgb, dim=True)
    for i, b in enumerate(panels):
        _box_rect(ax, b, PALETTE[i % len(PALETTE)], f"panel {i}")
    ax.set_title(f"{len(panels)} panel(s) detected — verify, then digitize each "
                 "(one session per panel).", fontsize=9)
    return _save(fig, out_path)


def render_heatmap(rgb, grid_box, res, out_path):
    """Heatmap grid with each cell's sampled value annotated over the image."""
    fig, ax = _fig(rgb, dim=True)
    x, y, w, h = [float(v) for v in grid_box]
    _box_rect(ax, [x, y, w, h], "#ffcc00", "grid")
    nr, nc = res["n_rows"], res["n_cols"]
    for r in range(nr):
        for c in range(nc):
            cx, cy = res["centers"][r][c]
            ax.text(cx, cy, _fmt(res["matrix"][r][c]), color="#ff2020",
                    fontsize=6, ha="center", va="center")
    ax.set_title(f"Heatmap {nr}x{nc} — values mapped from the colorbar; verify "
                 "a few cells against the figure.", fontsize=9)
    return _save(fig, out_path)


def render_axis_labels(rgb, plot_box, x_labels, y_labels, out_path):
    """One image with x labels boxed/numbered (red, bottom) and y labels
    (blue, left), so the operator reads both axes' values in a single look."""
    x, y, w, h = [int(v) for v in plot_box]
    pad_l, pad_b = int(0.16 * w) + 30, int(0.14 * h) + 30
    crop = [max(0, x - pad_l), max(0, y - 10), w + pad_l + 14, h + pad_b + 14]
    fig, ax = _fig(crop_img := crop_region(rgb, crop), dim=True, figscale=9.0)
    ox, oy = crop[0], crop[1]
    _box_rect(ax, [x - ox, y - oy, w, h], "#bbbb00", None)
    for i, l in enumerate(x_labels):
        bx, by, bw, bh = l["bbox"]
        ax.add_patch(mpatches.Rectangle((bx - ox, by - oy), bw, bh, fill=False,
                                        edgecolor="#ff2020", lw=1.4))
        ax.text(bx - ox, by - oy - 2, str(i), color="#ff2020", fontsize=10, weight="bold")
    for i, l in enumerate(y_labels):
        bx, by, bw, bh = l["bbox"]
        ax.add_patch(mpatches.Rectangle((bx - ox, by - oy), bw, bh, fill=False,
                                        edgecolor="#2050ff", lw=1.4))
        ax.text(bx - ox - 2, by - oy, str(i), color="#2050ff", fontsize=10,
                weight="bold", ha="right")
    ax.set_title(f"x labels (red, L->R): {len(x_labels)}   |   y labels (blue, "
                 f"T->B): {len(y_labels)} — read values; pass to calibrate "
                 "--x-values/--y-values", fontsize=9)
    return _save(fig, out_path)


def crop_region(rgb, box):
    x, y, w, h = [int(v) for v in box]
    h0, w0 = rgb.shape[:2]
    return rgb[max(0, y):min(h0, y + h), max(0, x):min(w0, x + w)]


def render_ticks(rgb, axis, labels, band, out_path, scale=3):
    """Magnified axis strip with each detected label boxed and numbered, so the
    operator can read off the value for each index."""
    import cv2

    if band is None:
        band = [0, 0, rgb.shape[1], rgb.shape[0]]
    bx, by, bw, bh = [int(v) for v in band]
    pad = 12
    cx0, cy0 = max(0, bx - pad), max(0, by - pad)
    cx1 = min(rgb.shape[1], bx + bw + pad)
    cy1 = min(rgb.shape[0], by + bh + pad)
    crop = rgb[cy0:cy1, cx0:cx1]
    if crop.size == 0:
        crop = rgb
        cx0 = cy0 = 0
    big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    fig, ax = plt.subplots(figsize=(min(16, big.shape[1] / 90 + 2),
                                    min(16, big.shape[0] / 90 + 2)))
    ax.imshow(big)
    for i, l in enumerate(labels):
        x, y, w, h = l["bbox"]
        rx, ry = (x - cx0) * scale, (y - cy0) * scale
        ax.add_patch(mpatches.Rectangle((rx, ry), w * scale, h * scale, fill=False,
                                        edgecolor="#ff2020", lw=1.5))
        ax.text(rx, ry - 3, str(i), color="#ff2020", fontsize=11, weight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{axis}-axis labels — read each numbered box's value; pass them to "
                 f"--values in index order (0,1,2,...).", fontsize=9)
    return _save(fig, out_path)


def _fmt(v):
    v = float(v)
    if v == 0:
        return "0"
    av = abs(v)
    if av >= 1e4 or av < 1e-2:
        return f"{v:.2e}"
    if av >= 100:
        return f"{v:.0f}"
    if av >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"
