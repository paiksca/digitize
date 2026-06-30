# Operating guide for the LLM (Claude) driving `digitize`

You are the **operator**. This tool is a precision instrument; you are the
perception, semantics, and judgment around it. Read this before digitizing.

## The contract

**You own** (the tool will never do these — do not delegate them to it):
- Identifying the plot type, what each axis means, its unit, and its **scale**
  (linear vs `log10`). A wrong scale silently corrupts everything downstream.
- Reading tick **values** off the image (use `zoom` for small text).
- Mapping legend entries → series, including which color/shape is which.
- Looking at every overlay and deciding if it is right.

**The tool owns** (never report these from your own "vision" — you will be wrong):
- Exact pixel coordinates of ticks, markers, curves.
- Color segmentation, blob/curve detection, the coordinate transform, uncertainty.

When you need a pixel, give the tool an **approximate** location plus intent and
let it snap (`calibrate` snaps tick guesses; `extract --sample` reads the color
under your point). Never hand-type a final data value — extract it.

## Fast path (use this by default)

Built to minimize round-trips. Two commands get you to calibrated data:

```
digitize auto fig.png              # detects panels, plot boxes, AND both axes'
                                   # tick labels per panel; one combined overlay each
# For each panel session it prints: open overlays/axes.png, read the red (x) and
# blue (y) numbered labels, then ONE call:
digitize calibrate --session fig_p0.digitize \
   --x-values "0,8,16,24,...,72" --y-values "100,10,1" \
   --xscale linear --yscale log10
# then: manifest --series ... ; extract ... ; values --all ; verify ; fit
```

`auto` collapses panels+init+box+tick-localization into one call; the value-form
`calibrate` folds axis-scales + tick-values + transform-build into one more. A
**clipped/mis-detected tick is auto-rejected** (the result reports `dropped`), so
one bad label won't ruin the fit — but still check the calibration overlay.
`--xscale categorical` maps text labels (C1D1, C1D8, …) to 0,1,2,…. If a label
count looks wrong in the overlay, the `auto` JSON also lists `x_positions` /
`y_positions` so you can pass manual `--x-ref px=..,val=..` for the bad axis.

The numbered steps below are the manual/fallback reference for when the fast path
needs a hand (faint frames, hard axes).

## The protocol

0. **Multi-panel figure?** (A/B/C/D, stacked, grids — most biomed figures are.)
   Run `digitize panels IMAGE --init` first. It detects each panel's plot area,
   renders `IMAGE_panels.png` to verify, and creates one session per panel with
   the plot box already set. Then do the steps below per panel session. (If a
   panel's frame is too faint to detect, fall back to a manual `--plot-box`.)
   **Dense composite** (10+ varied sub-panels — box/scatter/heatmap mixed) where
   `panels`/`auto` under-detect? Run `digitize grid IMAGE` for a labelled pixel
   grid, read each sub-panel's box off it, and drive panels manually.

1. **`init <image>`** (single-panel) → note `plot_box_candidate` and
   `legend_candidates`. Open `overlays/overview.png`. Is the yellow box on the
   data area? If not, pass `--plot-box x,y,w,h` to `manifest`.

2. **Look, then `manifest`.** Decide plot type, axis names/units, and **scales**.
   If tick labels are tiny or ambiguous, `zoom --region x-axis` / `y-axis`
   first. Record series with their colors (read from the legend). Example:
   `--y "name=conc,unit=ng/mL,scale=log10"`.

3. **Calibrate — prefer the automatic path.** Reading tick *pixels* by eye is the
   single slowest, most error-prone step; let the tool localize them:
   - `digitize ticks --axis x` → open `overlays/ticks_x.png`; each label is boxed
     and numbered. Read the values and `ticks --axis x --values "0,8,16,..."`
     (left→right). Repeat `--axis y` (top→bottom; for log give "1000,100,10,1").
   - `digitize calibrate --auto` builds the transform from those. **This handles
     faint-gridline log axes that are otherwise the worst case.**
   - If a label count is wrong, the box is usually clipped by the **plot box** —
     fix `--plot-box` so it ends exactly at the axis line (a too-short box clips
     the end decades; a box ending above the axis makes the strip grab the axis
     line or on-axis data points). Re-run.
   - Hard axis (data points sitting on the axis, a subscript title hugging the
     labels)? Do the easy axis with `ticks`, give the hard one 2 points manually:
     `calibrate --auto --x-ref px=285,val=0 --x-ref px=1110,val=9000` (auto-y +
     manual-x mix).
   - Manual fallback for everything: `calibrate --x-ref px=..,val=.. --y-ref ...`
     (snapping moves rough guesses onto detected ticks).

   **Always then open `overlays/calibration.png`:** the green reprojected grid
   must sit **on** the figure's gridlines/ticks, and `reprojection_rms_px` should
   be **< ~1.5 px**. Cross-check one known value if the figure states one (a
   reference line, an annotated point) — e.g. a "353 ng/mL" dashed line should
   read ~353.

4. **`extract` one series at a time.** Identify the color first:
   - From the legend: `legend --box <legend_box>` returns each swatch's color.
   - Then `extract --series NAME --seed-color "#rrggbb"`, **or** point at one
     on-plot marker with `--sample px,py` (more robust to legend/plot color drift).
   - **Always `--exclude` an in-plot legend** (use the `legend_candidates` box).
   - Open `overlays/extraction.png`. Check for: points on gridline intersections
     (false positives), missed markers, merged markers (one circle where there
     are two). Points are **numbered** for editing.

5. **Correct** with `edit --series NAME --add px,py --remove i --move "i:px,py"`,
   or re-`extract` with better flags (see playbook below). Re-inspect.

6. **`values --all`** applies the transform → data + per-point uncertainty, and
   writes CSVs. **`verify`** renders a side-by-side round-trip
   (`overlays/verify.png`) — the reconstructed plot must look like the original
   crop — and prints `flags`. Resolve every flag.

7. **`fit`** when the data is clean: `exp1`/`exp2` (PK decay), `4pl`/`emax`
   (dose-response), `nca` (Cmax/Tmax/AUC/t½). Check R² and the fit overlay.

8. **`export`** the bundle.

## Playbook for finicky figures (your stated hazard: messy colors & legends)

| Symptom in the overlay | Fix |
|---|---|
| Stray points clustered top-right/corner | An in-plot legend — re-`extract --exclude <box>` |
| A chromatic series also grabbed gridlines/text | Lower `--tol` (e.g. 12); confirm `--seed-color` is the true color via `swatch --at` |
| Two series with near-identical colors bleed together | Seed each from its on-plot marker (`--sample`), not the legend; segmentation competes them in Lab space, so accurate seeds matter |
| Missed faint/small markers | Raise tolerance (`--tol 24`) or lower `--min-area` |
| One marker where there are two (dense cluster) | Lower `--split-factor` (e.g. 1.4); if still merged, `edit --add` the miss |
| Single fat marker split into two | Raise `--split-factor` or `--no-split` |
| Series differ by **shape, not color** (monochrome) | `extract --template x,y,w,h` on one exemplar marker |
| No usable legend | `palette --k N` to propose the dominant colors, then label them |
| Curve crosses another / doubps back | Tracing takes the longest run per column; if it jumps, restrict with `--roi` |

## Chart-type playbook

The framework is **primitive + composition**: nearly every plot is points,
curves, filled regions, bar segments, or whiskers in a calibrated coordinate
system. Identify the type, then use the matching `extract --kind` (or compose
primitives). Rich types attach their levels in each point's `extra` dict, which
`values` converts to data (keys `q1_y`, `q3_y`, `lo_x`, `hi_x`, …).

| Chart type | How to extract |
|---|---|
| Scatter / volcano / Bland-Altman | `--kind scatter` (seed color or `--template`); `--errorbars` (y) and/or `--errorbars-x` (x-CI) for bars; `--shape circle\|triangle\|diamond\|square` to split same-color series by marker shape |
| Dual-axis (e.g. concentration **left log** + efficacy **right linear**) | calibrate each axis to the series it belongs to — `ticks --axis y --side right` localizes the right axis (colored labels OK); make a session/calibration per axis if both are used |
| Line / time-course / PK / ROC | `--kind line` (`--smooth`, `--resample`, `--at`); semi-log → `yscale log10` |
| Shaded CI band | `--kind line --edge band` → y_lo/y_hi (and the fit line with `--edge center`) |
| Dose-response / exposure-response | line+band for the fit, `scatter`/`--errorbars` for binned points, then `fit 4pl|emax` |
| Kaplan-Meier survival | `--kind km` (monotone step trace); censor ticks / Guyot IPD = manual |
| Spaghetti (many trajectories) | trace the **summary** line (`--kind line`, seed its color); individuals rarely all-separable |
| Vertical bar / histogram | `--kind bar` |
| Horizontal bar | `--kind hbar` |
| Stacked bar | extract each color segment (`--kind hbar`/`bar` per color); use `width_px`/edges in `extra` |
| Waterfall (per-subject) | `--kind waterfall` per cohort color (sorted bar heights vs baseline) |
| Box plot | `--kind box [--orient v|h]` → q1/q3/median/whiskers |
| Forest plot | `--kind forest` → point estimate + CI per row (categorical y) |
| Violin | `--kind line --edge band` per group (outline width = density) |
| Heatmap | `digitize heatmap --grid BOX --rows N --cols M --colorbar BOX --vmin --vmax [--auto-grid]` → value matrix (cell color → value via colorbar; no x/y cal). Add `--auto-grid` (or `--col-edges`/`--row-edges`) when cells are unequally sized — e.g. cell-type blocks with different batch counts. Cells are sampled by dominant color, so gridlines don't dilute them. `--rows/--cols 0` auto-detects the count from the cell gridlines (the border color forms perpendicular-uniform lines; median spacing = cell size) — works whenever the heatmap has visible cell borders |
| Swimmer / timeline (per-subject bars) | `--kind hbar` once **per category color** (e.g. each WHO grade); each segment → row + start/end via `extra` edges. Event markers (onset, draws) via `scatter`/`--template`. Categorical y = subjects |
| Multi-Y-axis (e.g. mAU + pH + mS/cm) | one calibration **per axis**: calibrate+extract the series on its own axis, repeat for the next (separate sessions or recalibrate) |

**Categorical axes** (visit labels, dose groups, study names): `calibrate
--xscale categorical` (or `--yscale`) maps the labels to 0,1,2,…; names are
stored. Use it for the category axis of bar/box/forest/waterfall plots.

**A type not in the table?** Decompose it: calibrate the axes, then pull each
mark kind with the closest primitive (`scatter` for points, `line`/`--edge band`
for curves/regions, `bar`/`hbar` for rectangles, `--errorbars` for whiskers), and
read levels from `extra`. You (the agent) can see the figure — name the marks,
extract each, and assemble.

## Lessons from real figures (read these)

Validated on real biomedical screenshots; these patterns recur:

- **A fit line of the same color connects the markers** → blob/scatter detection
  collapses them into one component (or fragments with `--open`). Instead trace
  the curve: `--kind line` (the fit *is* the profile you'll model), or
  `--template` to match just the marker shape. `--open 3-5` strips a same-color
  line/whiskers from a scatter mask when you truly need discrete markers.
- **Faint log-axis gridlines** are the #1 error source. Snapping has nothing to
  lock onto, so the decade pixels you read by eye drive everything. `zoom` the
  decade *labels* and read their pixel centers off the ruler; pick two
  well-separated decades. A wrong decade spacing is a uniform multiplicative
  error in every value. The terminal slope (half-life) is offset-invariant, so
  it stays right even if the absolute scale is off — sanity-check absolute values
  against one point you can read confidently.
- **Shaded model bands / filled areas** (no markers): `--kind line --edge top`
  traces the upper envelope, `--edge band` captures BOTH envelopes as y_lo/y_hi
  (recover a CI band in one pass). Use a tight `--tol`; pastel fills sit close to
  white and need an accurate seed.
- **Error bars** are captured with `extract --errorbars` (scatter): each whisker
  is followed as the connected run through its marker, so neighbours / the
  connecting line / the axis don't corrupt it. A shaded band *behind* the markers
  still occludes the whisker — that case stays hard.
- **Achromatic (gray/black) series** can't be found by `palette` (it only sees
  chroma). Probe by lightness: data markers/lines are usually *darker* gray than
  the light gridlines, so a dark-gray seed + moderate `--tol` separates them.
- **Multi-panel figures**: digitize one panel per session (set `--plot-box` to
  that panel). Each panel has its own pixel→data calibration.
- **Always restrict to an ROI just inside the frame** (`--roi`) when tracing —
  the frame and axis are the same dark color and cause endpoint spikes.

## Good to know (robustness)

- **Every command returns JSON, including errors** (`{"error": ..., "kind": ...}`)
  — parse it; a non-empty `error` means the call failed, act on the message.
- **Calibration auto-rejects a bad tick** (Theil-Sen + outlier drop) and reports
  `dropped`; a non-zero count means one label was off — glance at the overlay.
- **Line/KM traces auto-remove isolated spikes** (stray same-color pixels in a
  column); real multi-column peaks survive. Pass `--no-declutter` to keep raw.
- **Boxes/ROIs are clipped to the image** automatically; out-of-range is safe.

## Discipline that prevents silent errors

- **Trust the overlay, not your pixel intuition.** If the green grid is off, the
  numbers are wrong no matter how plausible they look.
- **Scale before everything.** Re-read the y-axis: is it linear or log? On log
  axes, evenly-spaced gridlines are powers of ten.
- **A great fit does not validate extraction.** A 4PL will fit happily through
  points off by a constant calibration error (EC₅₀ shifts). Always pass `verify`
  first; treat `flags` as blocking.
- **Uncertainty is real signal.** `y_err` grows near the top of a log decade —
  if a point's error looks huge, it may sit where one pixel = a large value.
- **One series at a time, inspect, then the next.** Don't batch and hope.

## JSON fields to actually check

- `init`: `plot_box_candidate`, `legend_candidates`.
- `calibrate`: `reprojection_rms_px` (<1 good), `x_refs[].snapped` (did it snap?).
- `extract`: `n_points` (does it match the figure?), `color` (the color used).
- `verify`: `flags` (must be empty), per-series `n` and `x_monotonic`.
- `fit`: `r2`, and `params[...].ci95` (report CIs, not just point estimates).

## Quick reference

```
auto IMG [--single]              # FAST PATH: panels + boxes + both axes' ticks
calibrate --x-values ".." --y-values ".." --xscale .. --yscale ..   # one-shot
panels IMG [--init]              # multi-panel: one session per panel, box pre-set
grid IMG [--step N]              # labelled pixel grid to read boxes off composites
init IMG                         manifest --type --x --y --series
zoom --region x-axis|y-axis|BOX  legend --box BOX     palette --k N
swatch --at px,py                ticks --axis x|y [--side left|right] [--values "..."]
calibrate --auto                 calibrate --x-ref --y-ref [--no-snap] [--mode affine]
extract --series N [--seed-color|--sample|--template] [--exclude] [--tol]
        [--split-factor] [--open K] [--shape circle|triangle|diamond|square]
        [--roi] [--errorbars] [--errorbars-x]
        [--kind line --resample N|--at vals --smooth W --edge top|bottom|band]
heatmap --grid BOX --rows N --cols M --colorbar BOX --vmin V --vmax V [--log]
edit --series N [--add|--remove|--move]   values --all [--pixel-sigma]
verify    fit --series N --model 4pl|emax|exp1|exp2|nca    export    info
```
