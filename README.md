# digitize — an LLM-operated precision digitizer for biomedical graphs

A Python toolkit that an LLM (Claude Code / cowork) **operates** to extract
numerical data from raster figures — PK/PD curves, dose-response, survival
curves, scatter/line/bar plots — with calibrated uncertainty and PK/PD fits.

It is built on a strict division of labor:

| The LLM does (perception + judgment) | The tool does (deterministic precision) |
|---|---|
| Identify plot type, axes, units, scales | Sub-pixel tick snapping, coordinate transforms |
| Read tick **values**, map legend → series | Color-based series separation, curve tracing |
| Point at a marker / legend swatch (roughly) | Snap to the precise pixel; segment; find blobs |
| Inspect overlays, judge correctness, correct | Uncertainty propagation; round-trip checks; fits |

The LLM never types a data coordinate; the tool never guesses what an axis
*means*. They meet through a CLI (JSON out) and overlay PNGs designed to be read
by a vision model. **`AGENT_GUIDE.md` is the operating manual for the LLM.**

## Install

Requires Python ≥ 3.10.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .                   # add ".[dev]" for the test suite
```

Deps: numpy, scipy, opencv-python-headless, scikit-image, matplotlib, pillow, click.

```bash
pip install -e ".[dev]" && pytest  # 21 tests, all against synthetic ground truth
python tests/synth.py examples && python examples/run_demo.py   # end-to-end demo
```

## The operating loop

```bash
digitize init fig.png                       # detect plot box + legend candidates
digitize zoom --region y-axis               # read small tick labels precisely
digitize manifest --type pk \               # record what the LLM sees
  --x "name=time,unit=h,scale=linear" \
  --y "name=conc,unit=ng/mL,scale=log10" \
  --series "name=drugA,color=#1f77b4" --series "name=drugB,color=#d62728"
digitize ticks --axis x                     # auto-locate tick labels (read overlay)
digitize ticks --axis x --values "0,8,..."  # record values (left->right)
digitize ticks --axis y --values "100,10,1" # ...and y (top->bottom; logs OK)
digitize calibrate --auto                   # build transform from located ticks
# (or fully manual: calibrate --x-ref px=75,val=0 --y-ref py=400,val=1 ...)
digitize extract --series drugA --sample 149,139 --exclude 445,58,95,62
digitize values --all                       # -> data + per-point uncertainty
digitize verify                             # round-trip overlay + flags
digitize fit --series drugA --model exp1    # PK/PD fit with CIs
digitize export
```

Every command prints JSON and (where useful) writes an overlay PNG to
`<session>/overlays/`. After each step the LLM looks at the overlay and either
proceeds or corrects (`digitize edit`, re-`extract` with different flags, etc.).

## Commands

`auto` · `panels` · `grid` · `init` · `manifest` · `zoom` · `legend` ·
`palette` · `swatch` · `ticks` · `calibrate` · `extract` · `heatmap` · `edit` ·
`values` · `verify` · `fit` · `export` · `info` (run `digitize <cmd> --help`).

**`auto` — the agentic fast path.** This tool is operated by an AI agent, so
round-trips are the scarce resource. `digitize auto IMAGE` detects panels, sets
each plot box, and localizes BOTH axes' tick labels per panel, emitting one
combined overlay each. The agent reads the numbered labels and runs a single
`digitize calibrate --x-values "..." --y-values "..." --xscale ... --yscale ...`
(scales + values + transform in one call; `categorical` maps text labels to
0,1,2,…). Calibration auto-rejects a mis-detected/clipped tick (reports
`dropped`) so one bad label can't ruin the fit. Two figure types beyond a panel —
e.g. a clipped log decade (m6) and a categorical visit axis like C1D1/C1D8 (m7) —
were validated this way to <0.25 px.

**`panels` — multi-panel auto-split.** Most biomedical figures are multi-panel.
`panels IMAGE --init` detects each panel's plot area (a bottom axis line meeting a
left axis line), renders a numbered overlay, and creates one ready-to-use session
per panel with its plot box pre-set — so you go straight to `ticks`/`extract`.

**`ticks` — automatic calibration.** The slow part of a real figure is locating
each tick to the pixel. `ticks --axis x/y` finds the label band beside an axis,
clusters it into per-tick positions, and renders a numbered overlay; you read the
values off it and pass them back with `--values`. `calibrate --auto` then builds
the transform. This turns a ~15-step manual calibration into 3 commands and works
on faint-gridline **log axes** (validated to ~1% on a real semi-log figure).

## Chart types

Built as **primitives + composition** so it adapts to nearly any plot. `extract
--kind`: `scatter` · `line` (+ `--edge band` for CI bands) · `bar` · `hbar` ·
`box [--orient v|h]` · `forest` · `km` (monotone survival step) · `waterfall`;
plus `digitize heatmap` (grid + colorbar → value matrix, `--auto-grid` for
unequal cells, dominant-color sampling so gridlines don't dilute) and
swimmer/timeline plots via `hbar` per category color. Multi-Y-axis plots: one
calibration per axis. `digitize grid IMAGE` gives a labelled pixel grid to read
sub-panel boxes off dense composites. Validated on a real 12-panel multi-omics
figure: the heatmap recovered the exact cell-lineage diagonal; only dendrograms
and embedding axes remain out of scope.

For **dense, overlapping, and occluded** multi-series figures, `digitize.extract.dense`
adds: distance-transform marker detection (`extract --kind markers`, true positions
under non-uniform sampling), continuity tracking + slope-fill, occluded
circle-center recovery from a partial arc, and momentum tracing of monochrome
overlapping lines with solid/dashed/dotted labelling — paired with exact-hex
nearest-color segmentation to separate near-identical shades. Validated on a
real siRNA PK/PD figure set (Patisiran, Revusiran, Givosiran, Inclisiran).
Rich marks attach their levels to each point's `extra` dict (box quartiles,
forest CI, bar edges), which `values` converts to data automatically. Axes can be
linear / log / logit / **categorical**. Anything not directly covered (violin,
heatmap, novel composites) is handled by composing primitives — see the
chart-type playbook in `AGENT_GUIDE.md`. Validated end-to-end on real KM curves
(recovered median 16.8 mo) and box/forest/band on synthetic ground truth.

## Handling messy figures

- **Many close colors / odd legends** — seed each series from the legend swatch
  (`legend --box ...` → `extract --seed-color`/`--sample`); segmentation uses a
  nearest-target rule in CIE Lab with background anchors so gridlines/text aren't
  vacuumed up. `palette` proposes colors when there is no usable legend.
- **In-plot legends** — `init` reports `legend_candidates`; pass the box to
  `extract --exclude` so swatches aren't read as data.
- **Dense / overlapping markers** — lower `extract --split-factor` to split
  merged blobs; or `edit --add` the misses you see in the overlay.
- **Shape-coded / monochrome series** — `extract --template x,y,w,h` matches a
  sample marker by shape instead of color.
- **Log axes** — set `scale=log10`; uncertainty correctly grows toward the top
  of the decade.

## Accuracy (synthetic ground truth)

`tests/synth.py` renders figures whose true data→pixel mapping is read from
matplotlib, so extraction error is measured against known answers. On the bundled
phantoms (`.venv/bin/python examples/run_demo.py`):

- **PK semilog**, two color-separated series: calibration RMS 0.38 px; mono-exp
  fit recovers t½ = 4.62 h (truth 4.62), R² = 0.999.
- **Dose-response**, log dose: 4PL recovers EC₅₀ ≈ 1.0 (truth 1.0), Hill ≈ 1.2
  (truth 1.2), R² > 0.999.
- **Line trace**: median y error < 0.15 over the data range.

```bash
.venv/bin/python -m pytest tests/ -q
```

## Layout

```
src/digitize/
  cli.py          # the operator surface (JSON out + overlays)
  imaging.py      # load, plot-box / panel / legend detection, masks
  ticks.py        # automatic tick-label localization
  calibrate.py    # tick detection, snapping, transform build
  transform.py    # pixel<->data (linear/log/logit/affine) + uncertainty
  color.py        # swatch sampling, palette, Lab nearest-target segmentation
  extract/        # scatter (blob+watershed+template), line/band, bar, errorbar
  overlay.py      # verification artifacts rendered for a vision model
  verify.py       # numeric round-trip / quality report
  fit.py          # 4PL, Emax/Hill, mono/bi-exponential, NCA
  session.py      # on-disk session state + provenance log
tests/            # synthetic ground-truth generator + accuracy tests
```
