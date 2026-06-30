"""Tunable constants. Kept in one place so behaviour is auditable and stable."""

# --- Uncertainty -----------------------------------------------------------
# Default 1-sigma pixel localization uncertainty (sub-pixel, from anti-aliasing
# and finite marker size). Propagated through the (possibly log) transform.
DEFAULT_PIXEL_SIGMA = 0.7

# --- Color segmentation ----------------------------------------------------
# Default CIEDE-ish LAB distance tolerance for assigning a pixel to a series.
DEFAULT_COLOR_TOL = 18.0
# A pixel is "background-ish" (axis/text/grid/whitespace) if its chroma is below
# this and we have no chromatic series claiming it. Chroma = sqrt(a^2 + b^2).
LOW_CHROMA = 10.0
# Pixels brighter than this (0-255) with low chroma are treated as paper white.
WHITE_L = 90.0

# --- Marker / blob detection ----------------------------------------------
# Connected components smaller than this many pixels are noise.
MIN_MARKER_AREA = 6
# Components larger than (median_area * this) are treated as merged markers and
# split via distance-transform watershed.
MERGE_SPLIT_FACTOR = 2.2
# Circularity below this is unlikely to be a discrete marker (4*pi*A / P^2).
MIN_CIRCULARITY = 0.25

# --- Tick detection --------------------------------------------------------
# Strip width (px) just outside the plot box that we scan for tick marks.
TICK_STRIP = 14
# Snap a supplied tick guess only if a detected candidate is within this radius.
TICK_SNAP_RADIUS = 12

# --- Rendering -------------------------------------------------------------
OVERLAY_DPI = 150
# Upscale factor for `zoom` crops so small tick labels are legible to the model.
ZOOM_SCALE = 4

# --- Plot-box detection ----------------------------------------------------
# Minimum fraction of the image dimension a line must span to count as an axis.
AXIS_LINE_MIN_FRAC = 0.35
