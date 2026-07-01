"""PK/PD model fitting on extracted data — the downstream payoff.

All regressions are weighted by the per-point extraction uncertainty when it is
available (``absolute_sigma`` so parameter CIs reflect real data error), and
report 95% CIs from the covariance plus R². NCA is non-compartmental (no fit):
linear-up/log-down trapezoidal AUC, terminal slope by best-R² log-linear window.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


# --- model functions -------------------------------------------------------
def four_pl(x, a, b, c, d):
    """4-parameter logistic. a: response as x->0, d: as x->inf, c: EC50, b: Hill."""
    return d + (a - d) / (1.0 + (x / c) ** b)


def emax_model(x, e0, emax, ec50, h):
    return e0 + emax * x**h / (ec50**h + x**h)


def mono_exp(t, c0, k):
    return c0 * np.exp(-k * t)


def bi_exp(t, A, alpha, B, beta):
    return A * np.exp(-alpha * t) + B * np.exp(-beta * t)


# --- core weighted fit -----------------------------------------------------
def _fit(func, x, y, p0, sigma=None, bounds=(-np.inf, np.inf)):
    use_sigma = None
    if sigma is not None:
        sigma = np.asarray(sigma, float)
        if np.all(np.isfinite(sigma)) and np.all(sigma > 0):
            use_sigma = sigma
    popt, pcov = curve_fit(func, x, y, p0=p0, sigma=use_sigma,
                           absolute_sigma=use_sigma is not None, bounds=bounds,
                           maxfev=40000)
    perr = np.sqrt(np.diag(pcov))
    yhat = func(x, *popt)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return popt, perr, r2, use_sigma is not None


def _params(names, popt, perr):
    out = {}
    for n, v, e in zip(names, popt, perr):
        out[n] = {"value": float(v), "se": float(e),
                  "ci95": [float(v - 1.96 * e), float(v + 1.96 * e)]}
    return out


def _curve(func, x, popt, log_x):
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if log_x and xmin > 0:
        xs = np.logspace(np.log10(xmin), np.log10(xmax), 100)
    else:
        xs = np.linspace(xmin, xmax, 100)
    return xs.tolist(), func(xs, *popt).tolist()


# --- public fits -----------------------------------------------------------
def fit_4pl(x, y, sigma=None):
    x = np.asarray(x, float); y = np.asarray(y, float)
    a0, d0 = float(y[np.argmin(x)]), float(y[np.argmax(x)])
    c0 = float(np.exp(np.mean(np.log(x[x > 0])))) if np.any(x > 0) else float(np.median(x))
    # use 3× the observed range as a buffer so partially-extracted curves (where
    # one plateau is outside the data range) are not clamped to a false EC50
    buf = max(3 * abs(d0 - a0), abs(a0) * 0.5, abs(d0) * 0.5, 10.0)
    lo_asymp, hi_asymp = min(a0, d0) - buf, max(a0, d0) + buf
    bounds = ([lo_asymp, 0.05, x[x > 0].min() * 1e-2 if np.any(x > 0) else 1e-9, lo_asymp],
              [hi_asymp, 20.0, x.max() * 1e2, hi_asymp])
    popt, perr, r2, w = _fit(four_pl, x, y, [a0, 1.0, c0, d0], sigma, bounds)
    p = _params(["resp_x0", "hill", "ec50", "resp_xinf"], popt, perr)
    xf, yf = _curve(four_pl, x, popt, log_x=True)
    return {"model": "4pl", "params": p, "derived": {"ec50": p["ec50"]["value"],
            "hill": p["hill"]["value"]}, "r2": r2, "weighted": w,
            "n": int(x.size), "curve": {"x": xf, "y": yf}}


def fit_emax(x, y, sigma=None):
    x = np.asarray(x, float); y = np.asarray(y, float)
    e0 = float(y[np.argmin(x)]); emax0 = float(y[np.argmax(x)] - e0)
    ec0 = float(np.median(x[x > 0])) if np.any(x > 0) else float(np.median(x))
    bounds = ([-np.inf, -np.inf, x[x > 0].min() * 1e-2 if np.any(x > 0) else 1e-9, 0.05],
              [np.inf, np.inf, x.max() * 1e2, 20.0])
    popt, perr, r2, w = _fit(emax_model, x, y, [e0, emax0, ec0, 1.0], sigma, bounds)
    p = _params(["e0", "emax", "ec50", "hill"], popt, perr)
    xf, yf = _curve(emax_model, x, popt, log_x=True)
    return {"model": "emax", "params": p, "derived": {"ec50": p["ec50"]["value"],
            "emax": p["emax"]["value"]}, "r2": r2, "weighted": w,
            "n": int(x.size), "curve": {"x": xf, "y": yf}}


def fit_exp1(t, c, sigma=None):
    t = np.asarray(t, float); c = np.asarray(c, float)
    c0 = float(c[np.argmin(t)])
    pos = c > 0
    k0 = 0.1
    if pos.sum() >= 2:
        sl = np.polyfit(t[pos], np.log(c[pos]), 1)[0]
        k0 = max(1e-4, -sl)
    popt, perr, r2, w = _fit(mono_exp, t, c, [c0, k0], sigma,
                             bounds=([0, 1e-6], [np.inf, np.inf]))
    p = _params(["c0", "k"], popt, perr)
    k = p["k"]["value"]
    xf, yf = _curve(mono_exp, t, popt, log_x=False)
    return {"model": "exp1", "params": p,
            "derived": {"half_life": float(np.log(2) / k) if k > 0 else None,
                        "auc_inf": float(p["c0"]["value"] / k) if k > 0 else None},
            "r2": r2, "weighted": w, "n": int(t.size), "curve": {"x": xf, "y": yf}}


def fit_exp2(t, c, sigma=None):
    t = np.asarray(t, float); c = np.asarray(c, float)
    cmax = float(np.max(c))
    p0 = [cmax * 0.6, 1.0, cmax * 0.4, 0.1]
    popt, perr, r2, w = _fit(bi_exp, t, c, p0, sigma,
                             bounds=([0, 1e-6, 0, 1e-6], [np.inf, np.inf, np.inf, np.inf]))
    p = _params(["A", "alpha", "B", "beta"], popt, perr)
    xf, yf = _curve(bi_exp, t, popt, log_x=False)
    return {"model": "exp2", "params": p, "r2": r2, "weighted": w,
            "n": int(t.size), "curve": {"x": xf, "y": yf}}


def nca(t, c):
    """Non-compartmental analysis. Returns Cmax, Tmax, AUClast/inf, lambda_z, t1/2."""
    t = np.asarray(t, float); c = np.asarray(c, float)
    order = np.argsort(t)
    t, c = t[order], c[order]
    imax = int(np.argmax(c))
    cmax, tmax = float(c[imax]), float(t[imax])

    auc = 0.0
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        c1, c2 = c[i], c[i + 1]
        if c2 < c1 and c1 > 0 and c2 > 0:  # log-down on the decline
            auc += dt * (c1 - c2) / np.log(c1 / c2)
        else:  # linear-up / flat
            auc += dt * (c1 + c2) / 2.0

    lam = t_half = auc_inf = None
    best_r2 = -np.inf
    tail_start = imax  # terminal phase is after Cmax
    idx = np.arange(tail_start, len(t))
    for start in range(tail_start, len(t) - 2):
        sel = idx[idx >= start]
        ts, cs = t[sel], c[sel]
        pos = cs > 0
        ts, cs = ts[pos], cs[pos]  # exclude BLQ/zero points within the window
        if ts.size < 3:
            continue
        slope, intercept = np.polyfit(ts, np.log(cs), 1)
        yhat = slope * ts + intercept
        ss_res = np.sum((np.log(cs) - yhat) ** 2)
        ss_tot = np.sum((np.log(cs) - np.mean(np.log(cs))) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else -np.inf
        if slope < 0 and r2 > best_r2:
            best_r2 = r2
            lam = float(-slope)
    if lam and lam > 0:
        t_half = float(np.log(2) / lam)
        auc_inf = float(auc + c[-1] / lam)

    return {"model": "nca", "derived": {
        "cmax": cmax, "tmax": tmax, "auc_last": float(auc),
        "lambda_z": lam, "half_life": t_half, "auc_inf": auc_inf,
        "lambda_z_r2": float(best_r2) if best_r2 > -np.inf else None},
        "n": int(t.size)}


FITTERS = {"4pl": fit_4pl, "emax": fit_emax, "hill": fit_emax,
           "exp1": fit_exp1, "exp2": fit_exp2, "nca": nca}
