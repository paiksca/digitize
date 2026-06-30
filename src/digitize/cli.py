"""Command-line surface. Each subcommand prints a JSON result on stdout (for the
LLM to parse) and, where useful, writes an overlay PNG (for the LLM to look at).

Typical operating loop:

    digitize init fig.png
    digitize manifest --type ... --x ... --y ... --series ...
    digitize calibrate --x-ref ... --y-ref ...      # inspect calib overlay
    digitize extract  --series A --sample 240,300   # inspect extraction overlay
    digitize values   --all                         # apply transform + uncertainty
    digitize verify                                 # round-trip
    digitize fit      --series A --model exp1
    digitize export
"""
from __future__ import annotations

import csv
from pathlib import Path

import click
import numpy as np

from . import color as colormod
from . import imaging, overlay, ticks as ticksmod, verify as verifymod
from .calibrate import build_calibration
from .extract import bar as barx
from .extract import boxplot as boxx
from .extract import forest as forestx
from .extract import heatmap as heatx
from .extract import km as kmx
from .extract import line as linex
from .extract import scatter as scatterx
from .extract.errorbar import extract_errorbars
from .fit import FITTERS
from .schemas import AxisSpec, DataPoint, Manifest, SeriesData, SeriesSpec
from .session import Session
from .util import (dumps, parse_box, parse_floats, parse_kv, parse_xy)


# --- helpers ---------------------------------------------------------------
def out(obj) -> None:
    click.echo(dumps(obj))


def resolve(session: str | None) -> Session:
    if session:
        return Session.open(session)
    cands = sorted(Path.cwd().glob("*.digitize"))
    if len(cands) == 1:
        return Session.open(cands[0])
    if not cands:
        raise click.ClickException("no *.digitize session in cwd; pass --session")
    raise click.ClickException(
        f"multiple sessions found ({[c.name for c in cands]}); pass --session")


def load_rgb_for(sess: Session):
    return imaging.load_rgb(sess.load_manifest().image)


def _norm_hex(c: str | None) -> str | None:
    if not c:
        return None
    return c if c.startswith("#") else "#" + c


def _series_render(sess: Session):
    """Build the per-series render payload (pixel points) for overlays."""
    rend = []
    for name in sess.list_series():
        sd = sess.load_series(name)
        rend.append({"name": sd.name, "color": sd.color, "kind": sd.kind,
                     "points": [{"px": p.px, "py": p.py} for p in sd.points]})
    return rend


def _value_refs(positions, values_str, coord, scale, manual_refs):
    """Build calibration refs by zipping auto-detected label positions with the
    operator's values. Categorical -> 0,1,2,... + stored names. Falls back to
    manual --x-ref/--y-ref when no values are given for that axis."""
    axis = coord[1]
    if values_str is None:
        refs = [{coord: float(parse_kv(r)[coord]), "val": float(parse_kv(r)["val"])}
                for r in manual_refs]
        if len(refs) < 2:
            raise click.ClickException(
                f"{axis}-axis: provide --{axis}-values (with `auto`) or >=2 --{axis}-ref")
        return refs, None
    raw = [v.strip() for v in values_str.split(",") if v.strip()]
    if scale == "categorical":
        names, vals = raw, [float(i) for i in range(len(raw))]
    else:
        names, vals = None, [float(v) for v in raw]
    if not positions or len(positions) != len(vals):
        raise click.ClickException(
            f"{axis}-axis: detected {len(positions or [])} labels but got {len(vals)} "
            "values. Open the axes overlay and pass one value per numbered box "
            "(or fix --plot-box, or use --{0}-ref).".format(axis))
    return [{coord: float(p), "val": v} for p, v in zip(positions, vals)], names


SESSION_OPT = click.option("--session", default=None,
                           help="session dir (default: the single *.digitize in cwd)")


@click.group()
@click.version_option(package_name="digitize")
def cli():
    """LLM-operated precision digitizer for biomedical graphs."""


# --- init ------------------------------------------------------------------
@cli.command()
@click.argument("image", type=click.Path(exists=True, dir_okay=False))
@click.option("--session", default=None, help="session dir (default: <image>.digitize)")
def init(image, session):
    """Load IMAGE, detect a plot-box candidate, start a session."""
    rgb = imaging.load_rgb(image)
    info = imaging.detect_plot_box(rgb)
    legends = imaging.detect_legend_candidates(rgb, info["plot_box"])
    sess = Session.for_image(image, session).ensure()
    m = Manifest(image=str(Path(image).resolve()), image_size=info["image_size"],
                 plot_box=info["plot_box"])
    sess.save_manifest(m)
    ov = overlay.render_overview(rgb, info, sess.overlays_dir / "overview.png",
                                 legends=legends)
    sess.log("init", {"image": image}, {"plot_box": info["plot_box"]})
    out({"session": str(sess.root), "image_size": info["image_size"],
         "plot_box_candidate": info["plot_box"],
         "vertical_lines": info["vertical_lines"],
         "horizontal_lines": info["horizontal_lines"],
         "legend_candidates": legends,
         "overview": ov,
         "next": "View the overview. Record what you see with `digitize manifest`. "
                 "If a legend sits inside the axes, plan to `extract --exclude` it."})


# --- auto (one-call setup for an agent) ------------------------------------
@cli.command()
@click.argument("image", type=click.Path(exists=True, dir_okay=False))
@click.option("--single", is_flag=True, help="treat as one panel (skip panel detection)")
@click.option("--prefix", default=None, help="session name prefix (default: image stem)")
def auto(image, single, prefix):
    """One call: detect panels, set each plot box, and localize BOTH axes' tick
    labels per panel. Emits one combined axes overlay per panel and creates a
    session each. Then per panel just read the overlay and run a single
    `digitize calibrate --x-values ... --y-values ... --xscale ... --yscale ...`.

    Collapses panels+init+box+tick-detection into one step — built for an agent
    minimizing round-trips."""
    rgb = imaging.load_rgb(image)
    h, w = rgb.shape[:2]
    boxes = ([imaging.detect_plot_box(rgb)["plot_box"]] if single
             else imaging.detect_panels(rgb))
    pre = prefix or Path(image).stem
    panels_out = []
    for i, box in enumerate(boxes):
        sess = Session(Path(image).parent / f"{pre}_p{i}.digitize").ensure()
        sess.save_manifest(Manifest(image=str(Path(image).resolve()),
                                    image_size=[w, h], plot_box=list(box)))
        xr = ticksmod.detect_axis_labels(rgb, box, "x")
        yr = ticksmod.detect_axis_labels(rgb, box, "y")
        sess.save_tick_positions({"x": [l["pos"] for l in xr["labels"]],
                                  "y": [l["pos"] for l in yr["labels"]]})
        ov = overlay.render_axis_labels(rgb, box, xr["labels"], yr["labels"],
                                        sess.overlays_dir / "axes.png")
        sess.log("auto", {"image": image, "panel": i}, {"box": list(box)})
        panels_out.append({
            "session": str(sess.root), "plot_box": list(box),
            "n_x_labels": len(xr["labels"]), "n_y_labels": len(yr["labels"]),
            # positions exposed so the agent can supply manual --x-ref/--y-ref
            # (px=<pos>,val=..) when a count looks wrong in the overlay
            "x_positions": [round(l["pos"], 1) for l in xr["labels"]],
            "y_positions": [round(l["pos"], 1) for l in yr["labels"]],
            "axes_overlay": ov})
    out({"n_panels": len(boxes), "panels": panels_out,
         "next": "Per panel: open its axes_overlay, read the red (x) and blue (y) "
                 "label values, then `digitize calibrate --session <S> "
                 '--x-values "..." --y-values "..." --xscale linear|log10|categorical '
                 "--yscale ...` (values in box-number order). Then `manifest "
                 "--series`, `extract`, `values`, `verify`, `fit`."})


# --- grid (coordinate overview for dense composites) -----------------------
@cli.command()
@click.argument("image", type=click.Path(exists=True, dir_okay=False))
@click.option("--step", default=200, type=int, help="grid spacing in pixels")
def grid(image, step):
    """Render the image with a labelled pixel grid. Use it to read sub-panel
    boxes off dense multi-panel composites where `panels`/`auto` under-detect."""
    rgb = imaging.load_rgb(image)
    out_path = Path(image).with_name(Path(image).stem + "_grid.png")
    ov = overlay.render_grid(rgb, out_path, step)
    out({"image_size": [rgb.shape[1], rgb.shape[0]], "step": step, "overlay": str(ov),
         "next": "Read each sub-panel's box (x,y,w,h) off the grid; pass via "
                 "manifest --plot-box (then ticks/extract) or heatmap --grid."})


# --- panels (multi-panel splitting) ----------------------------------------
@cli.command()
@click.argument("image", type=click.Path(exists=True, dir_okay=False))
@click.option("--init", "do_init", is_flag=True,
              help="create a ready-to-use session per panel (plot box pre-set)")
@click.option("--prefix", default=None, help="session name prefix (default: image stem)")
def panels(image, do_init, prefix):
    """Detect each panel's plot area in a multi-panel figure.

    With --init, spins up one session per panel with its plot box already set, so
    you can jump straight to `ticks`/`extract` on each."""
    rgb = imaging.load_rgb(image)
    boxes = imaging.detect_panels(rgb)
    h, w = rgb.shape[:2]
    ov = overlay.render_panels(rgb, boxes,
                               Path(image).with_name(Path(image).stem + "_panels.png"))
    result = {"n_panels": len(boxes), "panels": boxes, "overlay": str(ov)}
    if do_init:
        pre = prefix or Path(image).stem
        sessions = []
        for i, box in enumerate(boxes):
            sess = Session(Path(image).parent / f"{pre}_p{i}.digitize").ensure()
            sess.save_manifest(Manifest(image=str(Path(image).resolve()),
                                        image_size=[w, h], plot_box=list(box)))
            sessions.append(str(sess.root))
        result["sessions"] = sessions
        result["next"] = "per panel: set manifest scales/series, then ticks -> calibrate -> extract"
    else:
        result["next"] = "verify the overlay; re-run with --init to create per-panel sessions"
    out(result)


# --- manifest --------------------------------------------------------------
@cli.command()
@SESSION_OPT
@click.option("--type", "ptype", default=None, help="plot type, e.g. scatter/dose-response")
@click.option("--x", "xspec", default=None, help='axis spec "name=time,unit=h,scale=linear"')
@click.option("--y", "yspec", default=None, help='axis spec "name=conc,unit=ng/mL,scale=log10"')
@click.option("--series", "series", multiple=True,
              help='repeatable "name=A,color=#1f77b4,kind=scatter,marker=circle"')
@click.option("--plot-box", default=None, help='override "x,y,w,h"')
@click.option("--note", default=None)
def manifest(session, ptype, xspec, yspec, series, plot_box, note):
    """Record the semantic reading of the figure (what only the LLM can judge)."""
    sess = resolve(session)
    m = sess.load_manifest()
    if ptype:
        m.plot_type = ptype
    if xspec:
        kv = parse_kv(xspec)
        m.x = AxisSpec(kv.get("name", ""), kv.get("unit", ""), kv.get("scale", "linear"))
    if yspec:
        kv = parse_kv(yspec)
        m.y = AxisSpec(kv.get("name", ""), kv.get("unit", ""), kv.get("scale", "linear"))
    if series:
        specs = []
        for s in series:
            kv = parse_kv(s)
            specs.append(SeriesSpec(name=kv["name"], color=_norm_hex(kv.get("color")),
                                    kind=kv.get("kind", "scatter"),
                                    marker=kv.get("marker"), linestyle=kv.get("linestyle")))
        m.series = specs
    if plot_box:
        m.plot_box = list(parse_box(plot_box))
    if note is not None:
        m.notes = note
    sess.save_manifest(m)
    sess.log("manifest", {"type": ptype, "x": xspec, "y": yspec,
                          "series": list(series)}, {})
    out(m.to_dict())


# --- zoom ------------------------------------------------------------------
@cli.command()
@SESSION_OPT
@click.option("--region", required=True,
              help='"x-axis" | "y-axis" | "legend" | "full" | "x,y,w,h"')
@click.option("--scale", default=None, type=int, help="upscale factor")
def zoom(session, region, scale):
    """Magnified crop so small tick labels / legend text are legible."""
    from . import config

    sess = resolve(session)
    rgb = load_rgb_for(sess)
    m = sess.load_manifest()
    x, y, w, h = m.plot_box or imaging.detect_plot_box(rgb)["plot_box"]
    pad = max(8, int(0.04 * max(w, h)))
    presets = {
        "x-axis": [x - pad, y + h - pad, w + 2 * pad, int(h * 0.18) + 3 * pad],
        "y-axis": [x - int(w * 0.18) - 3 * pad, y - pad, int(w * 0.18) + 3 * pad, h + 2 * pad],
        "full": [0, 0, rgb.shape[1], rgb.shape[0]],
    }
    if region in presets:
        box = presets[region]
    elif region == "legend":
        raise click.ClickException("for 'legend', pass an explicit 'x,y,w,h' box")
    else:
        box = list(parse_box(region))
    sc = scale or config.ZOOM_SCALE
    path = overlay.save_zoom(rgb, box, sc, sess.overlays_dir / "zoom.png")
    out({"zoom": path, "region": box, "scale": sc})


# --- legend / palette / swatch (color discovery) ---------------------------
@cli.command()
@SESSION_OPT
@click.option("--box", required=True, help='legend region "x,y,w,h"')
def legend(session, box):
    """Auto-detect colored swatches in a legend region for the LLM to label."""
    sess = resolve(session)
    rgb = load_rgb_for(sess)
    sw = colormod.detect_legend_swatches(rgb, parse_box(box))
    ov = overlay.render_palette(sw, sess.overlays_dir / "legend.png",
                                title="Legend swatches (label each -> series)")
    sess.log("legend", {"box": box}, {"n": len(sw)})
    out({"swatches": sw, "overlay": ov,
         "hint": "Map each color to a series name, then pass to "
                 "`manifest --series` or `extract --seed-color`."})


@cli.command()
@SESSION_OPT
@click.option("--k", default=6, type=int)
def palette(session, k):
    """k-means the chromatic pixels in the plot area to propose series colors."""
    sess = resolve(session)
    rgb = load_rgb_for(sess)
    m = sess.load_manifest()
    pm = imaging.box_mask(rgb.shape, m.plot_box)
    pal = colormod.dominant_palette(rgb, pm, k=k)
    ov = overlay.render_palette(pal, sess.overlays_dir / "palette.png",
                                title="Dominant chromatic colors in plot area")
    out({"palette": pal, "overlay": ov})


@cli.command()
@SESSION_OPT
@click.option("--at", "at_", required=True, help='"px,py"')
@click.option("--radius", default=4, type=int)
def swatch(session, at_, radius):
    """Read the dominant color under a point (for `extract --seed-color`)."""
    sess = resolve(session)
    rgb = load_rgb_for(sess)
    px, py = parse_xy(at_)
    out(colormod.sample_swatch(rgb, px, py, radius=radius))


# --- ticks (auto label localization) ---------------------------------------
@cli.command()
@SESSION_OPT
@click.option("--axis", required=True, type=click.Choice(["x", "y"]))
@click.option("--values", default=None,
              help='comma-separated values in box order (left->right for x, '
                   'top->bottom for y), e.g. "0,8,16,24" or "1,10,100,1000"')
@click.option("--max-strip", default=90, type=int, help="how far outside the axis to scan")
@click.option("--dark", default=60.0, type=float,
              help="text darkness threshold (Lab L, 0-100; lower = stricter)")
@click.option("--side", default="left", type=click.Choice(["left", "right"]),
              help="y-axis: which side (right = a dual-axis plot's right axis)")
def ticks(session, axis, values, max_strip, dark, side):
    """Auto-locate axis tick LABELS; read their values off the overlay and pass
    them back via --values to record calibration references.

    For a dual-axis plot, localize the right axis with `--axis y --side right`
    (colored axis labels are detected, not just black ones)."""
    sess = resolve(session)
    rgb = load_rgb_for(sess)
    m = sess.load_manifest()
    res = ticksmod.detect_axis_labels(rgb, m.plot_box, axis, max_strip=max_strip,
                                      dark_thresh=dark, side=side)
    labels = res["labels"]
    ov = overlay.render_ticks(rgb, axis, labels, res["band"],
                              sess.overlays_dir / f"ticks_{axis}.png")
    result = {"axis": axis, "n_labels": len(labels),
              "positions": [round(l["pos"], 1) for l in labels], "overlay": ov}
    if values:
        vals = parse_floats(values)
        if len(vals) != len(labels):
            result["error"] = (f"detected {len(labels)} labels but got {len(vals)} "
                               "values. Inspect the overlay; adjust --max-strip/--dark "
                               "or pass exactly one value per numbered box.")
        else:
            coord = "px" if axis == "x" else "py"
            stored = [{coord: round(l["pos"], 2), "val": v}
                      for l, v in zip(labels, vals)]
            refs = sess.load_tick_refs()
            refs[axis] = stored
            sess.save_tick_refs(refs)
            result["stored"] = stored
            other = "y" if axis == "x" else "x"
            done = other in refs
            result["next"] = ("both axes recorded — run `digitize calibrate --auto`"
                              if done else f"now do the {other} axis, then "
                              "`digitize calibrate --auto`")
    else:
        result["next"] = (f"read each box's value, then `digitize ticks --axis {axis} "
                          '--values "..."` in box order')
    sess.log("ticks", {"axis": axis, "values": values}, {"n": len(labels)})
    out(result)


# --- calibrate -------------------------------------------------------------
@cli.command()
@SESSION_OPT
@click.option("--auto", is_flag=True, help="build from references recorded by `ticks`")
@click.option("--x-values", default=None, help='values for the x labels found by '
              '`auto`/`ticks`, in box order, e.g. "0,8,16" or "C1D1,C1D8" (categorical)')
@click.option("--y-values", default=None, help="values for the y labels, in box order")
@click.option("--xscale", default=None,
              type=click.Choice(["linear", "log10", "ln", "logit", "categorical"]))
@click.option("--yscale", default=None,
              type=click.Choice(["linear", "log10", "ln", "logit", "categorical"]))
@click.option("--x-ref", "x_refs", multiple=True, help='repeatable "px=120,val=0"')
@click.option("--y-ref", "y_refs", multiple=True, help='repeatable "py=455,val=1"')
@click.option("--snap/--no-snap", default=True, help="snap guesses to detected ticks")
@click.option("--mode", default="separable", type=click.Choice(["separable", "affine"]))
@click.option("--point", "points", multiple=True,
              help='affine mode, repeatable "px=..,py=..,x=..,y=.."')
def calibrate(session, auto, x_values, y_values, xscale, yscale, x_refs, y_refs,
              snap, mode, points):
    """Build the pixel<->data transform from tick references.

    With --auto, use the references recorded by `digitize ticks` (no snapping
    needed — those positions are already precise)."""
    sess = resolve(session)
    rgb = load_rgb_for(sess)
    m = sess.load_manifest()
    if mode == "affine":
        pts = [{k: float(v) for k, v in parse_kv(p).items()} for p in points]
        calib = build_calibration(rgb, m.plot_box, [], [], m.x.scale, m.y.scale,
                                  do_snap=False, mode="affine", affine_points=pts)
    else:
        scale_x, scale_y = m.x.scale, m.y.scale
        if x_values is not None or y_values is not None:
            pos = sess.load_tick_positions()
            sx = xscale or m.x.scale or "linear"
            sy = yscale or m.y.scale or "linear"
            xr, xnames = _value_refs(pos.get("x"), x_values, "px", sx, x_refs)
            yr, ynames = _value_refs(pos.get("y"), y_values, "py", sy, y_refs)
            scale_x = "linear" if sx == "categorical" else sx
            scale_y = "linear" if sy == "categorical" else sy
            m.x = AxisSpec(m.x.name, m.x.unit, scale_x, xnames)
            m.y = AxisSpec(m.y.name, m.y.unit, scale_y, ynames)
            sess.save_manifest(m)
            snap = False
        elif auto:
            refs = sess.load_tick_refs()
            # use recorded tick refs per axis; fall back to manual --x-ref/--y-ref
            xr = refs.get("x") or [{"px": float(parse_kv(r)["px"]),
                                    "val": float(parse_kv(r)["val"])} for r in x_refs]
            yr = refs.get("y") or [{"py": float(parse_kv(r)["py"]),
                                    "val": float(parse_kv(r)["val"])} for r in y_refs]
            if len(xr) < 2 or len(yr) < 2:
                raise click.ClickException(
                    "need >=2 refs per axis: run `digitize ticks` for each axis, or "
                    "supply --x-ref/--y-ref for the one the localizer missed")
            snap = False
        else:
            xr = [{"px": float(parse_kv(r)["px"]), "val": float(parse_kv(r)["val"])}
                  for r in x_refs]
            yr = [{"py": float(parse_kv(r)["py"]), "val": float(parse_kv(r)["val"])}
                  for r in y_refs]
        calib = build_calibration(rgb, m.plot_box, xr, yr, scale_x, scale_y,
                                  do_snap=snap, mode="separable")
    sess.save_calibration(calib)
    ov = overlay.render_calibration(rgb, m.plot_box, calib,
                                    sess.overlays_dir / "calibration.png")
    sess.log("calibrate", {"mode": mode, "snap": snap}, calib.get("transform"))
    res = {"reprojection_rms_px": calib["reprojection_rms_px"],
           "transform": calib["transform"], "overlay": ov}
    if mode == "separable":
        res["x_refs"] = calib["x_refs"]
        res["y_refs"] = calib["y_refs"]
        dropped = calib["transform"].get("dropped", {"x": 0, "y": 0})
        nd = dropped.get("x", 0) + dropped.get("y", 0)
        res["dropped_outliers"] = dropped
        note = "Inspect the green grid against the figure's."
        if nd:
            note = (f"Auto-dropped {nd} outlier tick ref(s) {dropped} (likely a "
                    "mis-detected label or a clipped decade). " + note)
        res["note"] = note
    out(res)


# --- extract ---------------------------------------------------------------
@cli.command()
@SESSION_OPT
@click.option("--series", "sname", required=True)
@click.option("--kind", default=None, type=click.Choice(
    ["scatter", "line", "bar", "box", "forest", "hbar", "km", "waterfall"]))
@click.option("--orient", default="v", type=click.Choice(["v", "h"]),
              help="box/bar orientation (v=vertical, h=horizontal)")
@click.option("--baseline", default=None, type=float,
              help="bar/waterfall baseline pixel (default: axis edge)")
@click.option("--seed-color", default=None, help="hex color for this series")
@click.option("--sample", "samples", multiple=True, help='repeatable "px,py" to read color')
@click.option("--template", default=None, help='"x,y,w,h" exemplar marker crop (shape match)')
@click.option("--roi", default=None, help='restrict to "x,y,w,h"')
@click.option("--exclude", "excludes", multiple=True, help='repeatable "x,y,w,h" to mask out')
@click.option("--tol", default=None, type=float, help="LAB color tolerance")
@click.option("--min-area", default=None, type=int)
@click.option("--max-area", default=None, type=int)
@click.option("--no-split", is_flag=True, help="don't split merged markers")
@click.option("--split-factor", default=None, type=float,
              help="split blobs bigger than this x median area (lower=denser markers)")
@click.option("--open", "open_k", default=1, type=int,
              help="morphological opening kernel; raise to 3-5 to strip same-color "
                   "fit lines / error-bar whiskers so only marker cores remain")
@click.option("--shape", default=None,
              type=click.Choice(["circle", "triangle", "diamond", "square"]),
              help="scatter: keep only this marker outline (splits same-color series "
                   "drawn with different shapes; 'circle' also drops error-bar caps)")
@click.option("--resample", default=None, type=int, help="line: N evenly-spaced points")
@click.option("--at", "at_vals", default=None, help="line: sample at data x-values "
              '"1,2,4,8" (needs calibration)')
@click.option("--smooth", default=0, type=int, help="line: moving-average window")
@click.option("--edge", default="center", type=click.Choice(["center", "top", "bottom", "band"]),
              help="line: trace longest-run center (default), top/bottom envelope, or "
                   "'band' to capture both envelopes of a shaded region (-> y_lo/y_hi)")
@click.option("--no-declutter", is_flag=True,
              help="line: keep the raw trace (don't auto-remove isolated spikes)")
@click.option("--errorbars", is_flag=True, help="scatter: also measure vertical (y) whiskers")
@click.option("--errorbars-x", is_flag=True,
              help="scatter: also measure horizontal (x) whiskers, e.g. a time CI")
@click.option("--match-threshold", default=0.6, type=float)
def extract(session, sname, kind, orient, baseline, seed_color, samples, template,
            roi, excludes, tol, min_area, max_area, no_split, split_factor, open_k,
            shape, resample, at_vals, smooth, edge, no_declutter, errorbars,
            errorbars_x, match_threshold):
    """Extract one series in pixel space (color-seeded or template-matched)."""
    sess = resolve(session)
    rgb = load_rgb_for(sess)
    m = sess.load_manifest()
    spec = m.series_by_name(sname)
    kind = kind or (spec.kind if spec else "scatter")
    color_hex = _norm_hex(seed_color) or (spec.color if spec else None)

    plot_mask = imaging.box_mask(rgb.shape, m.plot_box)
    excl = [parse_box(b) for b in excludes]
    plot_mask = imaging.subtract_boxes(plot_mask, excl)
    if roi:
        plot_mask &= imaging.box_mask(rgb.shape, parse_box(roi))

    pts: list[dict] = []
    used_color = color_hex
    if template:
        tb = parse_box(template)
        tmpl = imaging.crop(rgb, tb)
        pts = scatterx.match_markers(rgb, tmpl, threshold=match_threshold,
                                     roi_mask=plot_mask)
    else:
        # determine the color model
        if samples:
            labs = []
            for s in samples:
                px, py = parse_xy(s)
                labs.append(colormod.sample_swatch(rgb, px, py)["rgb"])
            rgb_mean = np.mean(labs, axis=0)
            cmodel = colormod.SeriesColorModel.from_rgb(sname, rgb_mean)
            used_color = cmodel.hex
        elif color_hex:
            from .util import hex_to_rgb
            cmodel = colormod.SeriesColorModel.from_rgb(sname, hex_to_rgb(color_hex))
        else:
            raise click.ClickException(
                "need --seed-color, --sample, or --template to identify the series")
        masks = colormod.segment_series(rgb, plot_mask, [cmodel], tol=tol)
        smask = masks[sname]
        if kind == "scatter":
            pts = scatterx.extract_markers(smask, min_area=min_area, max_area=max_area,
                                           split_merged=not no_split,
                                           split_factor=split_factor, open_k=open_k,
                                           shape=shape)
            if errorbars or errorbars_x:
                ink = imaging.ink_mask(rgb) | smask
                eax = ("both" if errorbars and errorbars_x
                       else "x" if errorbars_x else "y")
                pts = extract_errorbars(ink, pts, m.plot_box, axis=eax)
        elif kind == "line":
            at_px = None
            if at_vals:
                if not sess.has_calibration():
                    raise click.ClickException("--at needs calibration first")
                t = sess.transform()
                ym = m.plot_box[1] + m.plot_box[3] / 2
                at_px = [float(t.data_to_pixel(v, t.pixel_to_data(0, ym)[1])[0])
                         for v in parse_floats(at_vals)]
            pts = linex.trace_curve(smask, m.plot_box, resample=resample,
                                    at_px=at_px, smooth=smooth, edge=edge,
                                    declutter=not no_declutter)
        elif kind in ("bar", "waterfall"):
            bars = barx.extract_bars(smask, m.plot_box, baseline=baseline, orient="v")
            pts = [{"px": b["px"], "py": b["py"], "extra": {"base_py": b["py_base"]}}
                   for b in bars]
        elif kind == "hbar":
            bars = barx.extract_bars(smask, m.plot_box, baseline=baseline, orient="h")
            pts = [{"px": b["px"], "py": b["py"],
                    "extra": {"left_px": b["px_left"], "right_px": b["px_right"]}}
                   for b in bars]
        elif kind == "box":
            bx = boxx.extract_boxes(smask, m.plot_box, orient=orient, min_area=min_area or 30)
            keys = (("q1_py", "q3_py", "median_py", "whis_lo_py", "whis_hi_py")
                    if orient == "v" else
                    ("q1_px", "q3_px", "median_px", "whis_lo_px", "whis_hi_px"))
            main = "median_py" if orient == "v" else "median_px"
            alt = "q3_py" if orient == "v" else "q3_px"
            pts = [{"px": b["px"] if orient == "v" else (b.get(main) or b[alt]),
                    "py": (b.get(main) or b[alt]) if orient == "v" else b["py"],
                    "extra": {k: b[k] for k in keys if b.get(k) is not None}}
                   for b in bx]
        elif kind == "forest":
            pts = [{"px": f["point_px"], "py": f["py"],
                    "extra": {"lo_px": f["lo_px"], "hi_px": f["hi_px"]}}
                   for f in forestx.extract_forest(smask, m.plot_box)]
        elif kind == "km":
            pts = kmx.extract_km(smask, m.plot_box, resample=resample)

    sd = SeriesData(name=sname, kind=kind, color=used_color,
                    points=[DataPoint(px=p["px"], py=p["py"], area=p.get("area"),
                                      score=p.get("score"), py_hi=p.get("py_hi"),
                                      py_lo=p.get("py_lo"), extra=p.get("extra"),
                                      source=p.get("source", "auto"))
                            for p in pts])
    sess.save_series(sd)
    ov = overlay.render_extraction(rgb, m.plot_box, _series_render(sess),
                                   sess.overlays_dir / "extraction.png", exclude=excl)
    sess.log("extract", {"series": sname, "kind": kind, "color": used_color,
                         "template": bool(template)}, {"n": len(pts)})
    sample = [{"i": i, "px": round(p["px"], 1), "py": round(p["py"], 1)}
              for i, p in enumerate(pts[:12])]
    out({"series": sname, "kind": kind, "color": used_color, "n_points": len(pts),
         "overlay": ov, "sample": sample,
         "next": "Inspect the overlay; fix with `digitize edit`, then `digitize values`."})


# --- heatmap ---------------------------------------------------------------
@cli.command()
@SESSION_OPT
@click.option("--grid", required=True, help='heatmap cell grid region "x,y,w,h"')
@click.option("--rows", default=0, type=int, help="number of rows (0 = auto-detect)")
@click.option("--cols", default=0, type=int, help="number of columns (0 = auto-detect)")
@click.option("--colorbar", required=True, help='colorbar region "x,y,w,h"')
@click.option("--vmin", required=True, type=float)
@click.option("--vmax", required=True, type=float)
@click.option("--cbar-orient", default="v", type=click.Choice(["v", "h"]))
@click.option("--cbar-reverse", is_flag=True, help="vmin/vmax ends are swapped")
@click.option("--log", "log_scale", is_flag=True, help="log color scale")
@click.option("--auto-grid", is_flag=True,
              help="auto-detect cell boundaries from color changes (unequal cells)")
@click.option("--col-edges", default=None, help='explicit column boundaries "x0,x1,..." (cols+1)')
@click.option("--row-edges", default=None, help='explicit row boundaries "y0,y1,..." (rows+1)')
def heatmap(session, grid, rows, cols, colorbar, vmin, vmax, cbar_orient,
            cbar_reverse, log_scale, auto_grid, col_edges, row_edges):
    """Read a heatmap: sample each grid cell's color and map it to a value via
    the colorbar (no x/y calibration needed). Returns the value matrix + CSV.

    Use --auto-grid (or --col-edges/--row-edges) for heatmaps whose cells aren't
    evenly sized (e.g. cell-type blocks with different batch counts)."""
    sess = resolve(session)
    rgb = load_rgb_for(sess)
    gbox = parse_box(grid)
    res = heatx.extract_heatmap(rgb, gbox, rows, cols, parse_box(colorbar),
                                vmin, vmax, cbar_orient, cbar_reverse, log_scale,
                                col_edges=parse_floats(col_edges) if col_edges else None,
                                row_edges=parse_floats(row_edges) if row_edges else None,
                                auto_grid=auto_grid)
    cpath = sess.root / "heatmap.csv"
    with open(cpath, "w", newline="") as f:
        w = csv.writer(f)
        for row in res["matrix"]:
            w.writerow([round(v, 4) for v in row])
    from .util import write_json
    write_json(sess.root / "heatmap.json", res["matrix"])
    ov = overlay.render_heatmap(rgb, gbox, res, sess.overlays_dir / "heatmap.png")
    sess.log("heatmap", {"grid": grid, "rows": rows, "cols": cols}, {})
    out({"n_rows": res["n_rows"], "n_cols": res["n_cols"],
         "matrix_first_rows": res["matrix"][:3], "csv": str(cpath), "overlay": ov,
         "next": "Verify a few cells in the overlay against the figure's colors "
                 "(if rows/cols were auto-detected, check the count is right)."})


# --- edit (manual touch-up) ------------------------------------------------
@cli.command()
@SESSION_OPT
@click.option("--series", "sname", required=True)
@click.option("--add", "adds", multiple=True, help='repeatable "px,py"')
@click.option("--remove", "removes", multiple=True, type=int, help="repeatable index")
@click.option("--move", "moves", multiple=True, help='repeatable "idx:px,py"')
def edit(session, sname, adds, removes, moves):
    """Manually add / remove / move points the auto-extractor got wrong."""
    sess = resolve(session)
    rgb = load_rgb_for(sess)
    m = sess.load_manifest()
    sd = sess.load_series(sname)
    pts = list(sd.points)
    for mv in moves:
        idx_s, xy = mv.split(":", 1)
        i = int(idx_s)
        px, py = parse_xy(xy)
        pts[i].px, pts[i].py, pts[i].source = px, py, "moved"
        pts[i].x = pts[i].y = pts[i].x_err = pts[i].y_err = None
    for i in sorted({int(r) for r in removes}, reverse=True):
        if 0 <= i < len(pts):
            pts.pop(i)
    for a in adds:
        px, py = parse_xy(a)
        pts.append(DataPoint(px=px, py=py, source="manual"))
    sd.points = pts
    sess.save_series(sd)
    ov = overlay.render_extraction(rgb, m.plot_box, _series_render(sess),
                                   sess.overlays_dir / "extraction.png")
    sess.log("edit", {"series": sname, "add": list(adds), "remove": list(removes),
                      "move": list(moves)}, {"n": len(pts)})
    out({"series": sname, "n_points": len(pts), "overlay": ov})


# --- values (apply calibration) --------------------------------------------
@cli.command()
@SESSION_OPT
@click.option("--series", "sname", default=None)
@click.option("--all", "do_all", is_flag=True)
@click.option("--pixel-sigma", default=None, type=float, help="1-sigma pixel error")
def values(session, sname, do_all, pixel_sigma):
    """Apply the calibration to extracted pixels -> data values + uncertainty."""
    from . import config

    sess = resolve(session)
    if not sess.has_calibration():
        raise click.ClickException("no calibration; run `digitize calibrate` first")
    t = sess.transform()
    sig = config.DEFAULT_PIXEL_SIGMA if pixel_sigma is None else pixel_sigma
    names = sess.list_series() if do_all else [sname]
    if not names or names == [None]:
        raise click.ClickException("pass --series NAME or --all")
    results = []
    for name in names:
        sd = sess.load_series(name)
        for p in sd.points:
            x, y = t.pixel_to_data(p.px, p.py)
            xe, ye = t.uncertainty(p.px, p.py, sig, sig)
            p.x, p.y = float(x), float(y)
            p.x_err, p.y_err = float(xe), float(ye)
            if p.py_hi is not None:
                _, yhi = t.pixel_to_data(p.px, p.py_hi)
                _, ylo = t.pixel_to_data(p.px, p.py_lo)
                p.y_hi, p.y_lo = float(yhi), float(ylo)
                p.y_err = float(abs(yhi - ylo) / 2.0)
            if p.extra:  # convert rich-mark pixel levels (box/forest/bar) to data
                conv = {}
                for k, v in list(p.extra.items()):
                    if v is None:
                        continue
                    if k.endswith("_py"):
                        conv[k[:-3] + "_y"] = float(t.pixel_to_data(p.px, v)[1])
                    elif k.endswith("_px"):
                        conv[k[:-3] + "_x"] = float(t.pixel_to_data(v, p.py)[0])
                p.extra.update(conv)
        sess.save_series(sd)
        csv_path = _write_csv(sess, sd)
        results.append({"series": name, "n": len(sd.points), "csv": csv_path,
                        "sample": [{"x": round(p.x, 4), "y": round(p.y, 4),
                                    "y_err": round(p.y_err, 4) if p.y_err else None}
                                   for p in sd.points[:8]]})
    sess.log("values", {"series": names, "pixel_sigma": sig}, {"n_series": len(results)})
    out({"pixel_sigma": sig, "series": results})


def _write_csv(sess: Session, sd: SeriesData) -> str:
    path = sess.series_dir / f"{sd.name}.csv"
    cols = ["x", "y", "x_err", "y_err", "y_lo", "y_hi", "px", "py", "source"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for p in sd.points:
            w.writerow([getattr(p, c) for c in cols])
    return str(path)


# --- verify ----------------------------------------------------------------
@cli.command()
@SESSION_OPT
def verify(session):
    """Round-trip overlay + numeric quality report."""
    sess = resolve(session)
    rgb = load_rgb_for(sess)
    m = sess.load_manifest()
    calib = sess.load_calibration() if sess.has_calibration() else {}
    series_list = [sess.load_series(n) for n in sess.list_series()]
    ov = overlay.render_verify(rgb, m.plot_box, m, series_list,
                               sess.overlays_dir / "verify.png")
    rep = verifymod.report(m, calib, series_list)
    rep["overlay"] = ov
    sess.log("verify", {}, {"flags": rep["flags"]})
    out(rep)


# --- fit -------------------------------------------------------------------
@cli.command()
@SESSION_OPT
@click.option("--series", "sname", required=True)
@click.option("--model", required=True, type=click.Choice(list(FITTERS)))
def fit(session, sname, model):
    """Fit a PK/PD model (4pl, emax/hill, exp1, exp2) or run NCA."""
    sess = resolve(session)
    m = sess.load_manifest()
    sd = sess.load_series(sname)
    x = np.array([p.x for p in sd.points if p.x is not None], float)
    y = np.array([p.y for p in sd.points if p.y is not None], float)
    if x.size < 3:
        raise click.ClickException(f"need >=3 calibrated points (have {x.size}); "
                                   "run `digitize values` first")
    fitter = FITTERS[model]
    if model == "nca":
        res = fitter(x, y)
    else:
        ye = [p.y_err for p in sd.points if p.x is not None]
        sigma = np.array(ye, float) if all(e for e in ye) else None
        res = fitter(x, y, sigma=sigma)
    from .util import write_json
    write_json(sess.fits_dir / f"{sname}.{model}.json", res)
    ov = overlay.render_fit(sd, res, m, sess.overlays_dir / f"fit_{sname}.png")
    res_out = {k: v for k, v in res.items() if k != "curve"}
    res_out["overlay"] = ov
    sess.log("fit", {"series": sname, "model": model}, res.get("derived", {}))
    out(res_out)


# --- export ----------------------------------------------------------------
@cli.command()
@SESSION_OPT
@click.option("--format", "fmt", default="both", type=click.Choice(["csv", "json", "both"]))
@click.option("--out", "outpath", default=None)
def export(session, fmt, outpath):
    """Bundle manifest + calibration + all series (+ fits) for downstream use."""
    from .util import read_json, write_json

    sess = resolve(session)
    m = sess.load_manifest()
    series_list = [sess.load_series(n) for n in sess.list_series()]
    bundle = {
        "manifest": m.to_dict(),
        "calibration": sess.load_calibration() if sess.has_calibration() else None,
        "series": [sd.to_dict() for sd in series_list],
        "fits": {p.stem: read_json(p) for p in sorted(sess.fits_dir.glob("*.json"))},
    }
    written = {}
    if fmt in ("json", "both"):
        jpath = outpath if (outpath and fmt == "json") else str(sess.root / "export.json")
        write_json(jpath, bundle)
        written["json"] = jpath
    if fmt in ("csv", "both"):
        cpath = str(sess.root / "export_long.csv")
        with open(cpath, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["series", "x", "y", "x_err", "y_err", "y_lo", "y_hi"])
            for sd in series_list:
                for p in sd.points:
                    w.writerow([sd.name, p.x, p.y, p.x_err, p.y_err, p.y_lo, p.y_hi])
        written["csv"] = cpath
    sess.log("export", {"format": fmt}, written)
    out({"written": written, "n_series": len(series_list)})


# --- info ------------------------------------------------------------------
@cli.command()
@SESSION_OPT
def info(session):
    """Summarize the session state."""
    sess = resolve(session)
    m = sess.load_manifest()
    calib = sess.load_calibration() if sess.has_calibration() else None
    series = []
    for n in sess.list_series():
        sd = sess.load_series(n)
        series.append({"name": n, "kind": sd.kind, "n_points": len(sd.points),
                       "calibrated": any(p.x is not None for p in sd.points)})
    out({"session": str(sess.root), "plot_type": m.plot_type,
         "x": m.x.to_dict(), "y": m.y.to_dict(), "plot_box": m.plot_box,
         "calibrated": calib is not None,
         "reprojection_rms_px": calib.get("reprojection_rms_px") if calib else None,
         "series": series,
         "fits": [p.name for p in sorted(sess.fits_dir.glob("*.json"))]})


def main():
    """Entry point. Any error — usage or unexpected — is emitted as JSON on
    stdout (never a raw traceback) so an agent always gets parseable output."""
    import sys

    try:
        cli.main(standalone_mode=False)
    except click.ClickException as e:
        out({"error": e.format_message(), "kind": "usage"})
        sys.exit(2)
    except click.exceptions.Abort:
        sys.exit(130)
    except SystemExit as e:  # --help / --version printed normally
        sys.exit(e.code if e.code is not None else 0)
    except FileNotFoundError as e:
        out({"error": str(e), "kind": "missing_file"})
        sys.exit(1)
    except Exception as e:  # noqa: BLE001 - surface everything as JSON
        out({"error": str(e), "kind": type(e).__name__})
        sys.exit(1)


if __name__ == "__main__":
    main()
