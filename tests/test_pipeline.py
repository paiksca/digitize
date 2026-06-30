"""End-to-end accuracy tests against synthetic ground truth."""
from __future__ import annotations

import numpy as np
import pytest

from digitize import imaging
from digitize.calibrate import build_calibration
from digitize.color import SeriesColorModel, segment_series
from digitize.extract.line import trace_curve
from digitize.extract.scatter import extract_markers
from digitize.fit import fit_4pl
from digitize.transform import AxisTransform
from digitize.util import hex_to_rgb

import synth


@pytest.fixture(scope="module")
def figs(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    return synth.generate_all(d)


def _calib(truth):
    x_refs = [{"px": t["px"], "val": t["val"]} for t in truth["xticks"]]
    y_refs = [{"py": t["px"], "val": t["val"]} for t in truth["yticks"]]
    rgb = imaging.load_rgb(truth["image"])
    calib = build_calibration(rgb, truth["plot_box"], x_refs, y_refs,
                              truth["scale_x"], truth["scale_y"], do_snap=False)
    return AxisTransform.from_dict(calib["transform"]), calib


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-9)


@pytest.mark.parametrize("key", ["pk", "dose", "line"])
def test_calibration_recovers_truth(figs, key):
    truth = figs[key]
    t, calib = _calib(truth)
    assert calib["reprojection_rms_px"] < 1.0
    # every true data pixel should map back to its true value
    for s in truth["series"]:
        for p in s["points"]:
            x, y = t.pixel_to_data(p["px"], p["py"])
            assert _rel(float(x), p["x"]) < 0.02 or abs(float(x) - p["x"]) < 0.2
            assert _rel(float(y), p["y"]) < 0.02


def test_pk_two_series_separation_and_accuracy(figs):
    truth = figs["pk"]
    t, _ = _calib(truth)
    rgb = imaging.load_rgb(truth["image"])
    pm = imaging.box_mask(rgb.shape, truth["plot_box"])
    models = [SeriesColorModel.from_rgb(s["name"], hex_to_rgb(s["color"]))
              for s in truth["series"]]
    masks = segment_series(rgb, pm, models)
    for s in truth["series"]:
        pts = extract_markers(masks[s["name"]])
        truth_px = np.array([[p["px"], p["py"]] for p in s["points"]])
        assert abs(len(pts) - len(s["points"])) <= 1, s["name"]
        # match each extracted point to nearest truth, compare in data space
        rel_errs = []
        for q in pts:
            d = np.hypot(truth_px[:, 0] - q["px"], truth_px[:, 1] - q["py"])
            j = int(np.argmin(d))
            x, y = t.pixel_to_data(q["px"], q["py"])
            rel_errs.append(_rel(float(y), s["points"][j]["y"]))
        assert np.median(rel_errs) < 0.05, (s["name"], np.median(rel_errs))


def test_dose_response_fit_recovers_ec50(figs):
    truth = figs["dose"]
    t, _ = _calib(truth)
    rgb = imaging.load_rgb(truth["image"])
    pm = imaging.box_mask(rgb.shape, truth["plot_box"])
    s = truth["series"][0]
    model = SeriesColorModel.from_rgb(s["name"], hex_to_rgb(s["color"]))
    mask = segment_series(rgb, pm, [model])[s["name"]]
    pts = extract_markers(mask)
    xs, ys = [], []
    for q in pts:
        x, y = t.pixel_to_data(q["px"], q["py"])
        xs.append(float(x)); ys.append(float(y))
    res = fit_4pl(np.array(xs), np.array(ys))
    ec50 = res["derived"]["ec50"]
    assert 0.4 < ec50 < 2.5, ec50  # truth EC50 = 1.0
    assert res["r2"] > 0.97


def test_line_trace_accuracy(figs):
    truth = figs["line"]
    t, _ = _calib(truth)
    rgb = imaging.load_rgb(truth["image"])
    pm = imaging.box_mask(rgb.shape, truth["plot_box"])
    s = truth["series"][0]
    model = SeriesColorModel.from_rgb(s["name"], hex_to_rgb(s["color"]))
    mask = segment_series(rgb, pm, [model])[s["name"]]
    pts = trace_curve(mask, truth["plot_box"])
    # convert traced pixels -> data, compare y at matching x to truth curve
    tx = np.array([p["x"] for p in s["points"]])
    ty = np.array([p["y"] for p in s["points"]])
    errs = []
    for q in pts:
        x, y = t.pixel_to_data(q["px"], q["py"])
        yt = np.interp(float(x), tx, ty)
        errs.append(abs(float(y) - yt))
    assert np.median(errs) < 0.15, np.median(errs)
