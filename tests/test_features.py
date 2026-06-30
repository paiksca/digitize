"""Accuracy tests for panel detection, tick localization, and error bars."""
from __future__ import annotations

import numpy as np
import pytest

from digitize import imaging
from digitize.calibrate import build_calibration
from digitize.color import SeriesColorModel, segment_series
from digitize.extract.errorbar import extract_errorbars
from digitize.extract.scatter import extract_markers
from digitize.ticks import detect_axis_labels
from digitize.transform import AxisTransform
from digitize.util import hex_to_rgb

import synth


@pytest.fixture(scope="module")
def figs(tmp_path_factory):
    return synth.generate_all(tmp_path_factory.mktemp("synth2"))


# --- panel detection -------------------------------------------------------
def test_detect_panels_grid(figs):
    truth = figs["panels"]
    rgb = imaging.load_rgb(truth["image"])
    panels = imaging.detect_panels(rgb)
    assert len(panels) == 4, f"expected 4 panels, got {len(panels)}"
    # every true panel has a detected box whose corner is within a few px
    for tb in truth["panels"]:
        assert any(abs(b[0] - tb[0]) < 8 and abs(b[1] - tb[1]) < 8
                   and abs(b[2] - tb[2]) < 12 and abs(b[3] - tb[3]) < 12
                   for b in panels), f"no panel matches {tb}"


# --- tick localization -----------------------------------------------------
def _match(detected, truth_px, tol=6):
    """Each truth tick has a detected label within tol; return match errors."""
    errs = []
    for t in truth_px:
        d = min(detected, key=lambda p: abs(p - t))
        errs.append(abs(d - t))
    return errs


@pytest.mark.parametrize("key,axis,scale", [
    ("errorbar", "x", "linear"), ("errorbar", "y", "linear"),
    ("dose", "x", "log10"), ("pk", "y", "log10")])
def test_tick_localizer(figs, key, axis, scale):
    truth = figs[key]
    rgb = imaging.load_rgb(truth["image"])
    res = detect_axis_labels(rgb, truth["plot_box"], axis)
    detected = [l["pos"] for l in res["labels"]]
    truth_px = [t["px"] for t in truth[f"{axis}ticks"]]
    assert len(detected) >= len(truth_px) - 1, (key, axis, len(detected), len(truth_px))
    errs = _match(detected, truth_px)
    assert np.median(errs) < 4, (key, axis, np.median(errs))


# --- error bars ------------------------------------------------------------
def test_errorbars_recover_sd(figs):
    truth = figs["errorbar"]
    rgb = imaging.load_rgb(truth["image"])
    x_refs = [{"px": t["px"], "val": t["val"]} for t in truth["xticks"]]
    y_refs = [{"py": t["px"], "val": t["val"]} for t in truth["yticks"]]
    calib = build_calibration(rgb, truth["plot_box"], x_refs, y_refs, "linear",
                              "linear", do_snap=False)
    t = AxisTransform.from_dict(calib["transform"])

    box = truth["plot_box"]
    inset = [box[0] + 4, box[1] + 4, box[2] - 8, box[3] - 8]  # avoid the frame
    pm = imaging.box_mask(rgb.shape, inset)
    model = SeriesColorModel.from_rgb("m", hex_to_rgb("#000000"))
    mask = segment_series(rgb, pm, [model])["m"]

    markers = extract_markers(mask, open_k=3)
    assert abs(len(markers) - 7) <= 1, len(markers)
    bars = extract_errorbars(mask, markers, box)
    sds = []
    for b in bars:
        if b.get("py_hi") is None:
            continue
        _, y_hi = t.pixel_to_data(b["px"], b["py_hi"])
        _, y_lo = t.pixel_to_data(b["px"], b["py_lo"])
        sds.append(abs(float(y_hi) - float(y_lo)) / 2)
    assert len(sds) >= 5
    assert 4.0 < np.median(sds) < 8.5, np.median(sds)  # truth SD = 6


def test_robust_calibration_drops_clipped_tick():
    """One mis-detected tick (e.g. a decade clipped by a short plot box) must be
    auto-rejected so it can't poison the whole axis."""
    # 5 clean log decades at 68 px/decade + a 6th clipped 40 px too high
    y_refs = [(60.0, 10000), (128.0, 1000), (196.0, 100), (264.0, 10),
              (332.0, 1), (360.0, 0.1)]   # last should be ~400, not 360
    x_refs = [(100.0, 0), (300.0, 50), (500.0, 100)]
    t = AxisTransform.fit_separable(x_refs, y_refs, "linear", "log10")
    assert t.dropped["y"] == 1 and t.dropped["x"] == 0
    assert t.reprojection_rms_px < 1.5
    _, v = t.pixel_to_data(200.0, 60.0)
    assert abs(float(v) - 10000) / 10000 < 0.05
    # clean data must NOT trigger any drop
    t2 = AxisTransform.fit_separable(x_refs, y_refs[:5], "linear", "log10")
    assert t2.dropped["y"] == 0


def test_boxplot_quartiles(figs):
    from digitize.extract.boxplot import extract_boxes

    truth = figs["boxplot"]
    rgb = imaging.load_rgb(truth["image"])
    x_refs = [{"px": t["px"], "val": t["val"]} for t in truth["xticks"]]
    y_refs = [{"py": t["px"], "val": t["val"]} for t in truth["yticks"]]
    t = AxisTransform.from_dict(build_calibration(
        rgb, truth["plot_box"], x_refs, y_refs, "linear", "linear",
        do_snap=False)["transform"])
    pm = imaging.box_mask(rgb.shape, truth["plot_box"])
    mask = segment_series(rgb, pm, [SeriesColorModel.from_rgb(
        "b", hex_to_rgb("#4a90d9"))])["b"]
    boxes = extract_boxes(mask, truth["plot_box"], orient="v")
    assert len(boxes) == 2
    for b, st in zip(boxes, truth["boxes"]):
        g = lambda py: float(t.pixel_to_data(b["px"], py)[1])  # noqa: E731
        assert abs(g(b["q1_py"]) - st["q1"]) < 3, (g(b["q1_py"]), st["q1"])
        assert abs(g(b["q3_py"]) - st["q3"]) < 3, (g(b["q3_py"]), st["q3"])
        assert b["median_py"] and abs(g(b["median_py"]) - st["med"]) < 3
        assert abs(g(b["whis_lo_py"]) - st["whislo"]) < 4
        assert abs(g(b["whis_hi_py"]) - st["whishi"]) < 4


def test_forest_point_and_ci(figs):
    from digitize.extract.forest import extract_forest

    truth = figs["forest"]
    rgb = imaging.load_rgb(truth["image"])
    x_refs = [{"px": t["px"], "val": t["val"]} for t in truth["xticks"]]
    y_refs = [{"py": t["px"], "val": t["val"]} for t in truth["yticks"]]
    t = AxisTransform.from_dict(build_calibration(
        rgb, truth["plot_box"], x_refs, y_refs, "linear", "linear",
        do_snap=False)["transform"])
    pm = imaging.box_mask(rgb.shape, truth["plot_box"])
    mask = segment_series(rgb, pm, [SeriesColorModel.from_rgb(
        "g", hex_to_rgb("#333333"))])["g"]
    rows = extract_forest(mask, truth["plot_box"])
    assert len(rows) == 3
    # rows come top-to-bottom (largest y first)
    studies = sorted(truth["studies"], key=lambda s: -s["row"])
    for r, st in zip(rows, studies):
        est = float(t.pixel_to_data(r["point_px"], r["py"])[0])
        lo = float(t.pixel_to_data(r["lo_px"], r["py"])[0])
        hi = float(t.pixel_to_data(r["hi_px"], r["py"])[0])
        assert abs(est - st["est"]) < 0.12, (est, st["est"])
        assert abs(lo - st["lo"]) < 0.12 and abs(hi - st["hi"]) < 0.12


def test_km_monotone_trace(figs):
    from digitize.extract.km import extract_km

    truth = figs["km"]
    rgb = imaging.load_rgb(truth["image"])
    x_refs = [{"px": t["px"], "val": t["val"]} for t in truth["xticks"]]
    y_refs = [{"py": t["px"], "val": t["val"]} for t in truth["yticks"]]
    t = AxisTransform.from_dict(build_calibration(
        rgb, truth["plot_box"], x_refs, y_refs, "linear", "linear",
        do_snap=False)["transform"])
    pm = imaging.box_mask(rgb.shape, truth["plot_box"])
    mask = segment_series(rgb, pm, [SeriesColorModel.from_rgb(
        "s", hex_to_rgb("#c0392b"))])["s"]
    pts = extract_km(mask, truth["plot_box"], resample=40)
    xs = [float(t.pixel_to_data(p["px"], p["py"])[0]) for p in pts]
    ys = [float(t.pixel_to_data(p["px"], p["py"])[1]) for p in pts]
    assert all(ys[i + 1] <= ys[i] + 0.02 for i in range(len(ys) - 1))  # non-increasing
    tx = [d["x"] for d in truth["series"][0]["points"]]
    ty = [d["y"] for d in truth["series"][0]["points"]]
    errs = [abs(y - np.interp(x, tx, ty)) for x, y in zip(xs, ys)]
    assert np.median(errs) < 0.05


def test_heatmap_values(figs):
    from digitize.extract.heatmap import extract_heatmap

    truth = figs["heatmap"]
    rgb = imaging.load_rgb(truth["image"])
    res = extract_heatmap(rgb, truth["grid_box"], truth["n_rows"], truth["n_cols"],
                          truth["colorbar_box"], truth["vmin"], truth["vmax"], "v")
    got = np.array(res["matrix"])
    exp = np.array(truth["matrix"])
    assert got.shape == exp.shape
    assert np.median(np.abs(got - exp)) < 0.8, np.median(np.abs(got - exp))
    # --auto-grid (boundary auto-detection) must also recover it
    res2 = extract_heatmap(rgb, truth["grid_box"], truth["n_rows"], truth["n_cols"],
                           truth["colorbar_box"], truth["vmin"], truth["vmax"], "v",
                           auto_grid=True)
    assert np.median(np.abs(np.array(res2["matrix"]) - exp)) < 1.2
    # gridline-based auto-count of rows/cols
    from digitize.extract.heatmap import _count_cells
    assert _count_cells(rgb, truth["grid_box"], "row") == truth["n_rows"]
    assert _count_cells(rgb, truth["grid_box"], "col") == truth["n_cols"]
    # and the full pipeline with rows/cols=0 (auto) recovers the matrix
    res3 = extract_heatmap(rgb, truth["grid_box"], 0, 0, truth["colorbar_box"],
                           truth["vmin"], truth["vmax"], "v")
    assert np.array(res3["matrix"]).shape == exp.shape


def test_swimmer_hbar(figs):
    from digitize.extract.bar import extract_bars

    truth = figs["swimmer"]
    rgb = imaging.load_rgb(truth["image"])
    x_refs = [{"px": t["px"], "val": t["val"]} for t in truth["xticks"]]
    y_refs = [{"py": t["px"], "val": t["val"]} for t in truth["yticks"]]
    t = AxisTransform.from_dict(build_calibration(
        rgb, truth["plot_box"], x_refs, y_refs, "linear", "linear",
        do_snap=False)["transform"])
    pm = imaging.box_mask(rgb.shape, truth["plot_box"])
    mask = segment_series(rgb, pm, [SeriesColorModel.from_rgb(
        "g", hex_to_rgb("#2ca02c"))])["g"]  # the WHO-grade green segments
    bars = extract_bars(mask, truth["plot_box"], orient="h", min_area=10)
    # green appears in all 4 rows; row 3 spans days 0..18
    assert len(bars) >= 3
    spans = []
    for b in bars:
        row = round(float(t.pixel_to_data(b["px"], b["py"])[1]))
        a = float(t.pixel_to_data(b["px_left"], b["py"])[0])
        z = float(t.pixel_to_data(b["px_right"], b["py"])[0])
        spans.append((row, a, z))
    r3 = [s for s in spans if s[0] == 3]
    assert r3 and abs(r3[0][1] - 0) < 1 and abs(r3[0][2] - 18) < 1.5, spans


def test_shape_separation(figs):
    truth = figs["shapes"]
    rgb = imaging.load_rgb(truth["image"])
    roi = imaging.box_mask(rgb.shape, truth["plot_box"])
    cm = SeriesColorModel.from_rgb("b", hex_to_rgb("#1e8ffd"))
    smask = list(segment_series(rgb, roi, [cm], tol=24).values())[0]
    allm = extract_markers(smask, min_area=20)
    circ = extract_markers(smask, min_area=20, shape="circle")
    tri = extract_markers(smask, min_area=20, shape="triangle")
    assert len(circ) == 4, [p["shape"] for p in allm]   # 4 filled circles
    assert len(tri) == 3, [p["shape"] for p in allm]    # 3 open triangles
    cpx = sorted(p["px"] for p in circ)
    tpx = sorted(pt["px"] for pt in truth["series"][0]["points"])
    assert max(abs(a - b) for a, b in zip(cpx, tpx)) < 8


def test_declutter_removes_spikes_keeps_peaks():
    from digitize.extract.line import _hampel

    y = np.linspace(100.0, 200.0, 60)
    y[30] = 2000.0  # isolated stray-pixel spike
    out = _hampel(y)
    assert abs(out[30] - 150) < 12          # spike replaced by local median
    assert np.allclose(out[[0, 12, 45, 59]], y[[0, 12, 45, 59]])  # rest untouched
    # a real multi-column peak must survive (not be flattened)
    z = np.full(60, 100.0)
    z[27:33] = [140, 180, 200, 200, 180, 140]
    assert _hampel(z)[29] > 180


def test_band_envelopes(figs):
    from digitize.extract.line import trace_curve

    truth = figs["band"]
    rgb = imaging.load_rgb(truth["image"])
    x_refs = [{"px": t["px"], "val": t["val"]} for t in truth["xticks"]]
    y_refs = [{"py": t["px"], "val": t["val"]} for t in truth["yticks"]]
    calib = build_calibration(rgb, truth["plot_box"], x_refs, y_refs, "linear",
                              "linear", do_snap=False)
    t = AxisTransform.from_dict(calib["transform"])
    pm = imaging.box_mask(rgb.shape, truth["plot_box"])
    model = SeriesColorModel.from_rgb("b", hex_to_rgb("#6aa9d9"))
    mask = segment_series(rgb, pm, [model])["b"]
    pts = trace_curve(mask, truth["plot_box"], resample=20, edge="band")

    up = {p["x"]: p["y"] for p in truth["series"][0]["points"]}
    lo = {p["x"]: p["y"] for p in truth["series"][1]["points"]}
    ux, uy = np.array(list(up)), np.array(list(up.values()))
    lx, ly = np.array(list(lo)), np.array(list(lo.values()))
    err_hi, err_lo = [], []
    for p in pts:
        x, y_hi = t.pixel_to_data(p["px"], p["py_hi"])
        _, y_lo = t.pixel_to_data(p["px"], p["py_lo"])
        err_hi.append(abs(float(y_hi) - np.interp(float(x), ux, uy)))
        err_lo.append(abs(float(y_lo) - np.interp(float(x), lx, ly)))
    assert np.median(err_hi) < 0.4, np.median(err_hi)
    assert np.median(err_lo) < 0.4, np.median(err_lo)
