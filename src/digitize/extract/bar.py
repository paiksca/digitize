"""Bar extraction — vertical or horizontal, single or stacked.

Returns both edges of each bar so the caller can take the value edge (simple
bars), the height/width (stacked-segment bars), or the baseline. Waterfall plots
are just many vertical bars sorted along x; pass the whole color mask."""
from __future__ import annotations

from .common import clean_mask, components


def extract_bars(series_mask, plot_box, baseline=None, min_area: int = 20,
                 orient: str = "v") -> list[dict]:
    mask = clean_mask(series_mask, open_k=1, close_k=2)
    _, comps = components(mask, min_area=min_area)
    x0, y0, bw, bh = [int(v) for v in plot_box]
    bars = []
    for c in comps:
        bx, by, bw2, bh2 = c["bbox"]
        if orient == "v":
            base = float(baseline) if baseline is not None else float(y0 + bh)
            top, bottom = float(by), float(by + bh2)
            value = top if abs(top - base) > abs(bottom - base) else bottom
            bars.append({"px": c["cx"], "py": value, "py_base": base,
                         "py_top": top, "py_bottom": bottom, "height_px": float(bh2),
                         "source": "auto"})
        else:  # horizontal
            base = float(baseline) if baseline is not None else float(x0)
            left, right = float(bx), float(bx + bw2)
            value = right if abs(right - base) > abs(left - base) else left
            bars.append({"py": c["cy"], "px": value, "px_base": base,
                         "px_left": left, "px_right": right, "width_px": float(bw2),
                         "source": "auto"})
    bars.sort(key=lambda b: (b["py"], b["px"]) if orient == "h" else (b["px"], b["py"]))
    return bars
