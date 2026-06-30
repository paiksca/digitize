"""Tests for the dense/overlapping/occluded extraction techniques."""
from __future__ import annotations

import cv2
import numpy as np

from digitize.extract import dense


def test_detect_markers_finds_circles_ignores_line():
    mask = np.zeros((180, 220), np.uint8)
    centers = [(30, 40), (60, 95), (45, 150)]          # (col, row)
    for cx, cy in centers:
        cv2.circle(mask, (cx, cy), 5, 1, -1)
    cv2.line(mask, (40, 30), (95, 60), 1, 1)            # thin connector -> ignored
    pts = dense.detect_markers(mask.astype(bool), min_distance=8)
    assert len(pts) == 3
    found = sorted((round(p["px"]), round(p["py"])) for p in pts)
    assert found == sorted(centers)


def test_candidate_bands_two_bands():
    strip = np.zeros((50, 5), bool)
    strip[10:14] = True
    strip[30:34] = True
    bands = dense.candidate_bands(strip)
    assert len(bands) == 2
    assert abs(bands[0][0] - 11.5) < 1 and abs(bands[1][0] - 31.5) < 1


def test_track_series_rejects_crossing_distractor():
    xs = list(range(10))
    cands = [[(20.0, 8.0)] for _ in xs]
    cands[5] = [(60.0, 9.0), (20.0, 4.0)]   # distractor denser, real band present
    track = dense.track_series(cands, xs, tol_px=10)
    assert abs(track[5] - 20.0) < 3          # re-picked the on-trend band


def test_fill_gaps_slope_interpolates():
    out, est = dense.fill_gaps([10.0, None, 30.0], [0, 1, 2])
    assert abs(out[1] - 20.0) < 0.1 and est[1] is True


def test_recover_circle_from_occluded_arc():
    mask = np.zeros((60, 60), np.uint8)
    cv2.circle(mask, (30, 30), 6, 1, -1)
    mask[33:, :] = 0                         # hide the bottom half
    b = dense.recover_circle(mask.astype(bool), 30, 30, radius=6)
    assert b is not None and abs(b - 30) < 4


def test_trace_lines_through_crossing():
    mask = np.zeros((100, 100), bool)
    for x in range(100):
        mask[min(99, int(20 + 0.5 * x)), x] = True   # rising
        mask[max(0, int(70 - 0.5 * x)), x] = True     # falling
    tr = dense.trace_lines(mask, 10, [25, 65], 0, 99, 0, 99)
    ys = sorted(tr[k][90] for k in tr if 90 in tr[k])
    assert ys[1] - ys[0] > 20                 # lines stayed separated through the cross


def test_line_duty_cycle_solid_vs_dotted():
    solid = np.zeros((20, 100), bool); solid[10, :] = True
    dotted = np.zeros((20, 100), bool); dotted[10, ::3] = True
    tk = {x: 10 for x in range(100)}
    assert dense.line_duty_cycle(solid, tk, 0, 100) > 0.95
    assert dense.line_duty_cycle(dotted, tk, 0, 100) < 0.5
