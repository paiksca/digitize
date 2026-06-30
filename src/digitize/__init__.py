"""digitize — an LLM-operated precision digitizer for biomedical graphs.

The package is built around a strict division of labor:

* The **LLM** (Claude) provides perception and semantics: it identifies the plot
  type, reads axis labels / tick values / units, maps legend entries to series,
  and judges whether extraction looks right by inspecting overlay images.
* This **toolkit** provides deterministic precision: sub-pixel tick snapping,
  coordinate transforms (linear / log / logit / affine), color-based series
  separation, curve tracing, marker detection, uncertainty propagation, and
  PK/PD model fitting.

Nothing here guesses what an axis *means*; nothing the LLM does reports a raw
pixel coordinate. The two halves meet through the CLI and through overlay PNGs
that are rendered specifically to be inspected by a vision model.
"""

__version__ = "0.1.0"
