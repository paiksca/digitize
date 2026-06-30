"""Generate synthetic biomedical-style figures with EXACT ground truth.

Because we render with matplotlib, we can read the true data->pixel mapping from
``ax.transData`` and so know, to sub-pixel precision, where every data point and
tick lands in the saved PNG. That lets the tests measure real extraction error
against known answers — the closest thing to a calibrated phantom for a digitizer.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DPI = 160
FIGSIZE = (8.0, 6.0)  # -> 1280 x 960 px canvas (realistic resolution)


def _finish(fig, ax, path, scale_x, scale_y, series, xticks, yticks):
    """Save and compute the ground-truth pixel coordinates."""
    fig.savefig(path, dpi=DPI)
    fig.canvas.draw()
    W, H = fig.canvas.get_width_height()

    def to_px(x, y):
        disp = ax.transData.transform(np.column_stack([np.atleast_1d(x),
                                                        np.atleast_1d(y)]))
        return disp[:, 0], H - disp[:, 1]

    bbox = ax.get_window_extent(fig.canvas.get_renderer())
    plot_box = [float(bbox.x0), float(H - bbox.y1), float(bbox.width), float(bbox.height)]

    truth_series = []
    xs = ys = np.array([0.0])  # fallback for figures with no point-series
    for s in series:
        xs = np.array([p[0] for p in s["data"]], float)
        ys = np.array([p[1] for p in s["data"]], float)
        px, py = to_px(xs, ys)
        truth_series.append({
            "name": s["name"], "color": s["color"],
            "points": [{"x": float(a), "y": float(b), "px": float(c), "py": float(d)}
                       for a, b, c, d in zip(xs, ys, px, py)]})

    def ticks_px(vals, axis):
        out = []
        for v in vals:
            if axis == "x":
                p = to_px(v, ys.mean())[0][0]
            else:
                p = to_px(xs.mean(), v)[1][0]
            out.append({"val": float(v), "px": float(p)})
        return out

    truth = {
        "image": str(path), "image_size": [int(W), int(H)],
        "plot_box": plot_box, "scale_x": scale_x, "scale_y": scale_y,
        "xticks": ticks_px(xticks, "x"), "yticks": ticks_px(yticks, "y"),
        "series": truth_series,
    }
    plt.close(fig)
    return truth


def make_pk_semilog(path):
    """Two-compartment-ish PK: semilog concentration-time, two colored series."""
    t = np.array([0.5, 1, 2, 4, 6, 8, 12, 18, 24], float)
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    series = []
    for name, c0, k, color, marker in [("drugA", 100.0, 0.15, "#1f77b4", "o"),
                                        ("drugB", 60.0, 0.08, "#d62728", "s")]:
        conc = c0 * np.exp(-k * t)
        ax.semilogy(t, conc, marker, color=color, ms=7, ls="none", label=name)
        series.append({"name": name, "color": color,
                       "data": list(zip(t, conc))})
    ax.set_xlim(0, 25)
    ax.set_ylim(1, 200)
    ax.set_xlabel("time (h)")
    ax.set_ylabel("concentration (ng/mL)")
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax.grid(True, which="both", alpha=0.3)
    return _finish(fig, ax, path, "linear", "log10", series,
                   [0, 4, 8, 12, 16, 20, 24], [1, 10, 100])


def make_dose_response(path):
    """4PL dose-response on a log dose axis; single colored series."""
    dose = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100], float)
    top, bottom, ec50, hill = 100.0, 2.0, 1.0, 1.2
    resp = bottom + (top - bottom) / (1 + (dose / ec50) ** (-hill))
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.semilogx(dose, resp, "o", color="#2ca02c", ms=7, ls="none", label="compound")
    ax.set_xlim(0.005, 200)
    ax.set_ylim(0, 110)
    ax.set_xlabel("dose (uM)")
    ax.set_ylabel("response (%)")
    ax.grid(True, which="both", alpha=0.3)
    truth = _finish(fig, ax, path, "log10", "linear",
                    [{"name": "compound", "color": "#2ca02c",
                      "data": list(zip(dose, resp))}],
                    [0.01, 0.1, 1, 10, 100], [0, 25, 50, 75, 100])
    truth["params"] = {"ec50": ec50, "hill": hill, "top": top, "bottom": bottom}
    return truth


def make_line(path):
    """A single smooth colored curve for trace testing."""
    x = np.linspace(0, 10, 200)
    y = 5 + 3 * np.sin(0.6 * x) * np.exp(-0.1 * x)
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.plot(x, y, "-", color="#9467bd", lw=2, label="signal")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    step = 10
    return _finish(fig, ax, path, "linear", "linear",
                   [{"name": "signal", "color": "#9467bd",
                     "data": list(zip(x[::step], y[::step]))}],
                   [0, 2, 4, 6, 8, 10], [0, 2, 4, 6, 8, 10])


def make_errorbar(path):
    """Linear scatter with known, constant-SD error bars (no connecting line)."""
    t = np.array([1, 2, 4, 8, 16, 24, 32], float)
    y = 55.0 * (1 - np.exp(-t / 4.0))
    sd = 6.0
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.errorbar(t, y, yerr=sd, fmt="o", color="#000000", ms=6, capsize=4, ls="none")
    ax.set_xlim(0, 36)
    ax.set_ylim(0, 70)
    ax.set_xticks([0, 6, 12, 18, 24, 30, 36])
    ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70])
    truth = _finish(fig, ax, path, "linear", "linear",
                    [{"name": "m", "color": "#000000", "data": list(zip(t, y))}],
                    [0, 6, 12, 18, 24, 30, 36], [0, 10, 20, 30, 40, 50, 60, 70])
    truth["sd"] = sd
    return truth


def make_panels(path):
    """A 2x2 grid of framed plots, for panel-detection ground truth."""
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.0), dpi=DPI)
    x = np.linspace(0, 10, 60)
    for ax in axes.flat:
        ax.plot(x, np.sin(x), "k-")
        ax.set_xlim(0, 10)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    fig.canvas.draw()
    W, H = fig.canvas.get_width_height()
    boxes = []
    for ax in axes.flat:
        bb = ax.get_window_extent(fig.canvas.get_renderer())
        boxes.append([float(bb.x0), float(H - bb.y1), float(bb.width), float(bb.height)])
    plt.close(fig)
    return {"image": str(path), "image_size": [int(W), int(H)], "panels": boxes}


def make_band(path):
    """A shaded band with known upper and lower envelopes (for --edge band)."""
    x = np.linspace(0, 10, 120)
    lo = 2.0 + 0.4 * x
    hi = 6.0 + 0.4 * x
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.fill_between(x, lo, hi, color="#6aa9d9")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 15)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_yticks([0, 5, 10, 15])
    truth = _finish(fig, ax, path, "linear", "linear",
                    [{"name": "upper", "color": "#6aa9d9", "data": list(zip(x, hi))},
                     {"name": "lower", "color": "#6aa9d9", "data": list(zip(x, lo))}],
                    [0, 2, 4, 6, 8, 10], [0, 5, 10, 15])
    return truth


def make_boxplot(path):
    """Filled box plots with white median lines, known quartiles + whiskers."""
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    stats = [{"med": 40, "q1": 30, "q3": 50, "whislo": 15, "whishi": 68, "fliers": []},
             {"med": 55, "q1": 45, "q3": 65, "whislo": 25, "whishi": 88, "fliers": []}]
    bp = ax.bxp(stats, positions=[1, 2], widths=0.5, patch_artist=True,
                manage_ticks=False)
    for p in bp["boxes"]:
        p.set_facecolor("#4a90d9"); p.set_edgecolor("#3070b0")
    for wsk in bp["whiskers"] + bp["caps"]:
        wsk.set_color("#3070b0")
    for med in bp["medians"]:
        med.set_color("#ffffff"); med.set_linewidth(2)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    truth = _finish(fig, ax, path, "linear", "linear", [], [0, 1, 2, 3],
                    [0, 20, 40, 60, 80, 100])
    truth["boxes"] = stats
    return truth


def make_forest(path):
    """Horizontal forest plot: point estimate + CI per row (known values)."""
    studies = [(0, 1.0, 0.7, 1.4), (1, 1.8, 1.2, 2.6), (2, 0.6, 0.4, 0.9)]
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    for y, est, lo, hi in studies:
        ax.plot([lo, hi], [y, y], "-", color="#333333", lw=2)
        ax.plot([est], [y], "s", color="#333333", ms=9)
    ax.set_xlim(0, 3)
    ax.set_ylim(-0.6, 2.6)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_yticks([0, 1, 2])
    truth = _finish(fig, ax, path, "linear", "linear", [], [0, 1, 2, 3], [0, 1, 2])
    truth["studies"] = [{"row": y, "est": e, "lo": lo, "hi": hi}
                        for y, e, lo, hi in studies]
    return truth


def make_km(path):
    """Decreasing step (survival) curve with known levels."""
    t = [0, 2, 4, 6, 8, 10]
    s = [1.0, 0.82, 0.66, 0.52, 0.43, 0.4]
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.step(t, s, where="post", color="#c0392b", lw=2)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.0)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    truth = _finish(fig, ax, path, "linear", "linear",
                    [{"name": "s", "color": "#c0392b",
                      "data": list(zip(t, s))}], [0, 2, 4, 6, 8, 10],
                    [0, 0.25, 0.5, 0.75, 1.0])
    return truth


def make_heatmap(path):
    """A heatmap with white gridlines, a known high-contrast value matrix, and a
    vertical colorbar (viridis) — like a real heatmap (cells with a border)."""
    nr, nc = 5, 6
    matrix = np.array([[12.0 if (r + c) % 2 == 0 else 1.0 for c in range(nc)]
                       for r in range(nr)])
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    im = ax.pcolormesh(matrix, cmap="viridis", vmin=0, vmax=12,
                       edgecolors="white", linewidth=1.5)
    ax.invert_yaxis()  # row 0 at top, matching the extractor's reading order
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.04)
    fig.savefig(path, dpi=DPI)
    fig.canvas.draw()
    W, H = fig.canvas.get_width_height()

    def winbox(a):
        bb = a.get_window_extent(fig.canvas.get_renderer())
        return [float(bb.x0), float(H - bb.y1), float(bb.width), float(bb.height)]

    truth = {"image": str(path), "image_size": [int(W), int(H)],
             "grid_box": winbox(ax), "colorbar_box": winbox(cbar.ax),
             "matrix": matrix.tolist(), "vmin": 0, "vmax": 12,
             "n_rows": nr, "n_cols": nc}
    plt.close(fig)
    return truth


def make_swimmer(path):
    """Swimmer/timeline: per-row horizontal segments colored by category."""
    rows = {  # patient row -> list of (grade_color, start, end)
        0: [("#2ca02c", 0, 6), ("#fdae61", 6, 12)],
        1: [("#2ca02c", 0, 3), ("#d7301f", 3, 14)],
        2: [("#fdae61", 0, 8), ("#2ca02c", 8, 10)],
        3: [("#2ca02c", 0, 18)],
    }
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    for y, segs in rows.items():
        for color, a, b in segs:
            ax.barh(y, b - a, left=a, height=0.6, color=color)
    ax.set_xlim(0, 20)
    ax.set_ylim(-0.6, 3.6)
    ax.set_xticks([0, 4, 8, 12, 16, 20])
    ax.set_yticks([0, 1, 2, 3])
    truth = _finish(fig, ax, path, "linear", "linear", [], [0, 4, 8, 12, 16, 20],
                    [0, 1, 2, 3])
    truth["rows"] = {str(k): v for k, v in rows.items()}
    return truth


def make_shapes(path):
    """Two same-COLOR series distinguished only by marker SHAPE: filled circles
    and open triangles (tests shape-based series separation)."""
    cx = np.array([1.0, 2.0, 3.0, 4.0]); cy = np.array([2.0, 4.0, 3.0, 5.0])
    tx = np.array([1.5, 2.5, 3.5]); ty = np.array([5.0, 2.5, 4.0])
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.scatter(cx, cy, marker="o", s=170, c="#1e8ffd")
    ax.scatter(tx, ty, marker="^", s=220, facecolors="none", edgecolors="#1e8ffd",
               linewidths=2.2)
    ax.set_xlim(0, 5); ax.set_ylim(0, 6)
    ax.set_xticks([0, 1, 2, 3, 4, 5]); ax.set_yticks([0, 1, 2, 3, 4, 5, 6])
    truth = _finish(fig, ax, path, "linear", "linear",
                    [{"name": "circ", "color": "#1e8ffd", "data": list(zip(cx, cy))},
                     {"name": "tri", "color": "#1e8ffd", "data": list(zip(tx, ty))}],
                    [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5, 6])
    return truth


GENERATORS = {"pk": make_pk_semilog, "dose": make_dose_response, "line": make_line,
              "errorbar": make_errorbar, "panels": make_panels, "band": make_band,
              "boxplot": make_boxplot, "forest": make_forest, "km": make_km,
              "heatmap": make_heatmap, "swimmer": make_swimmer, "shapes": make_shapes}


def generate_all(outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    truths = {}
    for key, gen in GENERATORS.items():
        truths[key] = gen(outdir / f"{key}.png")
    return truths


if __name__ == "__main__":
    import sys
    from digitize.util import write_json

    d = Path(sys.argv[1] if len(sys.argv) > 1 else "examples")
    truths = generate_all(d)
    for k, t in truths.items():
        write_json(d / f"{k}.truth.json", t)
        print(f"{k}: {t['image']}  plot_box={[round(v) for v in t['plot_box']]}")
