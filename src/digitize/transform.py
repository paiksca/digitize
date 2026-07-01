"""Pixel <-> data coordinate transforms with uncertainty propagation.

Two modes:

* ``separable`` (the common case): axes are image-aligned. The x mapping depends
  only on the pixel column and the y mapping only on the pixel row. Each axis can
  be linear / log10 / ln / logit. Built from >=2 reference ticks per axis.
* ``affine``: axes may be rotated or sheared. Built from >=3 non-collinear
  reference points, each carrying both data coordinates.

In both modes the model is linear in a *transformed* coordinate ``u = scale.fwd(value)``
(e.g. ``u = log10(value)`` for a log axis). Uncertainty is propagated from a
1-sigma pixel localization error through the analytic Jacobian, so a point near
the top of a log axis correctly reports a larger error than one near the bottom.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_KINDS = ("linear", "log10", "ln", "logit")


class Scale:
    """A 1-D axis scale: maps data value <-> linear coordinate ``u``."""

    def __init__(self, kind: str = "linear"):
        if kind not in _KINDS:
            raise ValueError(f"unknown scale {kind!r}; choose from {_KINDS}")
        self.kind = kind

    def fwd(self, v):  # value -> u
        v = np.asarray(v, dtype=float)
        if self.kind == "linear":
            return v
        if self.kind == "log10":
            return np.log10(v)
        if self.kind == "ln":
            return np.log(v)
        return np.log(v / (1.0 - v))  # logit

    def inv(self, u):  # u -> value
        u = np.asarray(u, dtype=float)
        if self.kind == "linear":
            return u
        if self.kind == "log10":
            return 10.0**u
        if self.kind == "ln":
            return np.exp(u)
        return 1.0 / (1.0 + np.exp(-u))  # expit

    def dinv_du(self, u):  # d(value)/du at u
        u = np.asarray(u, dtype=float)
        if self.kind == "linear":
            return np.ones_like(u)
        if self.kind == "log10":
            return np.log(10.0) * (10.0**u)
        if self.kind == "ln":
            return np.exp(u)
        s = 1.0 / (1.0 + np.exp(-u))  # logit
        return s * (1.0 - s)


def _fit_line(p, u):
    """Least-squares u = m*p + b. Returns (m, b, rms_residual_in_u)."""
    p = np.asarray(p, float)
    u = np.asarray(u, float)
    A = np.column_stack([p, np.ones_like(p)])
    (m, b), *_ = np.linalg.lstsq(A, u, rcond=None)
    resid = u - (m * p + b)
    return float(m), float(b), float(np.sqrt(np.mean(resid**2)))


def _fit_line_robust(p, u):
    """Fit u = m*p + b, rejecting mis-detected ticks (e.g. a decade clipped by a
    slightly-short plot box). A Theil-Sen slope (median of pairwise slopes) gives
    an outlier-resistant line; points far from it are dropped, then a plain
    least-squares is refit on the inliers. Returns (m, b, kept_mask)."""
    p = np.asarray(p, float)
    u = np.asarray(u, float)
    if p.size <= 2:
        m, b, _ = _fit_line(p, u)
        return m, b, np.ones(p.size, bool)
    slopes = [(u[j] - u[i]) / (p[j] - p[i])
              for i in range(p.size) for j in range(i + 1, p.size)
              if p[j] != p[i]]
    if not slopes:  # all refs share the same pixel coordinate — degenerate
        m, b, _ = _fit_line(p, u)
        return m, b, np.ones(p.size, bool)
    m0 = float(np.median(slopes))
    b0 = float(np.median(u - m0 * p))
    resid = np.abs(u - (m0 * p + b0))
    span = max(float(u.max() - u.min()), 1e-9)
    mad = float(np.median(np.abs(resid - np.median(resid))))
    thresh = max(3 * 1.4826 * mad, 0.04 * span)
    keep = resid <= thresh
    if keep.sum() < 2:  # never drop below a fittable set
        keep = np.ones(p.size, bool)
    m, b, _ = _fit_line(p[keep], u[keep])
    return float(m), float(b), keep


@dataclass
class AxisTransform:
    mode: str  # "separable" | "affine"
    scale_x: str
    scale_y: str
    params: dict
    reprojection_rms_px: float
    n_refs: int

    # ---- builders ---------------------------------------------------------
    @classmethod
    def fit_separable(cls, x_refs, y_refs, scale_x="linear", scale_y="linear",
                      robust=True):
        """x_refs: list of (px, value); y_refs: list of (py, value)."""
        if len(x_refs) < 2 or len(y_refs) < 2:
            raise ValueError("separable mode needs >=2 x-refs and >=2 y-refs")
        sx, sy = Scale(scale_x), Scale(scale_y)
        xp = np.array([r[0] for r in x_refs], float)
        xu = np.asarray(sx.fwd([r[1] for r in x_refs]), float)
        yp = np.array([r[0] for r in y_refs], float)
        yu = np.asarray(sy.fwd([r[1] for r in y_refs]), float)
        if robust:
            mx, bx, xkeep = _fit_line_robust(xp, xu)
            my, by, ykeep = _fit_line_robust(yp, yu)
        else:
            mx, bx, _ = _fit_line(xp, xu)
            my, by, _ = _fit_line(yp, yu)
            xkeep = np.ones(xp.size, bool)
            ykeep = np.ones(yp.size, bool)
        params = {"mx": mx, "bx": bx, "my": my, "by": by}
        t = cls("separable", scale_x, scale_y, params, 0.0,
                int(xkeep.sum() + ykeep.sum()))
        xk = [r for i, r in enumerate(x_refs) if xkeep[i]]
        yk = [r for i, r in enumerate(y_refs) if ykeep[i]]
        t.reprojection_rms_px = t._reproj_separable(xk, yk)
        t.dropped = {"x": int((~xkeep).sum()), "y": int((~ykeep).sum())}
        return t

    @classmethod
    def fit_affine(cls, points, scale_x="linear", scale_y="linear"):
        """points: list of (px, py, x, y) — >=3, non-collinear."""
        if len(points) < 3:
            raise ValueError("affine mode needs >=3 calibration points")
        sx, sy = Scale(scale_x), Scale(scale_y)
        P = np.array([[p[0], p[1], 1.0] for p in points], float)
        ux = sx.fwd([p[2] for p in points])
        uy = sy.fwd([p[3] for p in points])
        cx, *_ = np.linalg.lstsq(P, ux, rcond=None)  # ux = a11 px + a12 py + t1
        cy, *_ = np.linalg.lstsq(P, uy, rcond=None)
        params = {
            "a11": float(cx[0]), "a12": float(cx[1]), "t1": float(cx[2]),
            "a21": float(cy[0]), "a22": float(cy[1]), "t2": float(cy[2]),
        }
        t = cls("affine", scale_x, scale_y, params, 0.0, len(points))
        t.reprojection_rms_px = t._reproj_affine(points)
        return t

    # ---- forward / inverse ------------------------------------------------
    def pixel_to_data(self, px, py):
        px = np.asarray(px, float)
        py = np.asarray(py, float)
        sx, sy = Scale(self.scale_x), Scale(self.scale_y)
        p = self.params
        if self.mode == "separable":
            ux = p["mx"] * px + p["bx"]
            uy = p["my"] * py + p["by"]
        else:
            ux = p["a11"] * px + p["a12"] * py + p["t1"]
            uy = p["a21"] * px + p["a22"] * py + p["t2"]
        return sx.inv(ux), sy.inv(uy)

    def data_to_pixel(self, x, y):
        sx, sy = Scale(self.scale_x), Scale(self.scale_y)
        ux = sx.fwd(np.asarray(x, float))
        uy = sy.fwd(np.asarray(y, float))
        p = self.params
        if self.mode == "separable":
            px = (ux - p["bx"]) / p["mx"]
            py = (uy - p["by"]) / p["my"]
            return px, py
        # invert the 2x2 linear part
        A = np.array([[p["a11"], p["a12"]], [p["a21"], p["a22"]]])
        Ainv = np.linalg.inv(A)
        du = np.stack([ux - p["t1"], uy - p["t2"]], axis=-1)  # (...,2)
        pix = du @ Ainv.T
        return pix[..., 0], pix[..., 1]

    def uncertainty(self, px, py, sigma_px, sigma_py):
        """1-sigma data-space uncertainty from pixel localization error."""
        px = np.asarray(px, float)
        py = np.asarray(py, float)
        sx, sy = Scale(self.scale_x), Scale(self.scale_y)
        p = self.params
        if self.mode == "separable":
            ux = p["mx"] * px + p["bx"]
            uy = p["my"] * py + p["by"]
            dx = np.abs(sx.dinv_du(ux) * p["mx"]) * sigma_px
            dy = np.abs(sy.dinv_du(uy) * p["my"]) * sigma_py
            return dx, dy
        ux = p["a11"] * px + p["a12"] * py + p["t1"]
        uy = p["a21"] * px + p["a22"] * py + p["t2"]
        gx = sx.dinv_du(ux)
        gy = sy.dinv_du(uy)
        dx = np.sqrt((gx * p["a11"] * sigma_px) ** 2 + (gx * p["a12"] * sigma_py) ** 2)
        dy = np.sqrt((gy * p["a21"] * sigma_px) ** 2 + (gy * p["a22"] * sigma_py) ** 2)
        return dx, dy

    # ---- reprojection error ----------------------------------------------
    def _reproj_separable(self, x_refs, y_refs):
        errs = []
        sx, sy = Scale(self.scale_x), Scale(self.scale_y)
        p = self.params
        for px, val in x_refs:
            pred = (sx.fwd(val) - p["bx"]) / p["mx"]
            errs.append(pred - px)
        for py, val in y_refs:
            pred = (sy.fwd(val) - p["by"]) / p["my"]
            errs.append(pred - py)
        return float(np.sqrt(np.mean(np.square(errs)))) if errs else 0.0

    def _reproj_affine(self, points):
        errs = []
        for px, py, x, y in points:
            ppx, ppy = self.data_to_pixel(x, y)
            errs.append(float(ppx) - px)
            errs.append(float(ppy) - py)
        return float(np.sqrt(np.mean(np.square(errs)))) if errs else 0.0

    # ---- serialization ----------------------------------------------------
    def to_dict(self):
        return {
            "mode": self.mode,
            "scale_x": self.scale_x,
            "scale_y": self.scale_y,
            "params": self.params,
            "reprojection_rms_px": self.reprojection_rms_px,
            "n_refs": self.n_refs,
            "dropped": getattr(self, "dropped", {"x": 0, "y": 0}),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            mode=d["mode"],
            scale_x=d["scale_x"],
            scale_y=d["scale_y"],
            params=d["params"],
            reprojection_rms_px=d.get("reprojection_rms_px", 0.0),
            n_refs=d.get("n_refs", 0),
        )
