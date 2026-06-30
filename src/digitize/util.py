"""Small shared helpers: color conversion, parsing of CLI mini-DSLs, JSON IO."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np


# --- color -----------------------------------------------------------------
def hex_to_rgb(s: str) -> tuple[int, int, int]:
    s = s.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        raise ValueError(f"bad hex color: {s!r}")
    return tuple(int(s[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def rgb_to_hex(rgb) -> str:
    r, g, b = (int(round(float(v))) for v in rgb[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


def rgb_to_lab(rgb) -> np.ndarray:
    """sRGB (0-255) -> CIE Lab (L in 0..100, a/b in ~-127..127).

    Accepts a single color (3,), a list of colors (N,3), or an image (H,W,3)
    and returns the same shape.
    """
    import cv2

    arr = np.asarray(rgb, dtype=np.float32)
    if arr.ndim == 1:  # single color
        img = arr.reshape(1, 1, 3)
        lab = cv2.cvtColor((img / 255.0).astype(np.float32), cv2.COLOR_RGB2LAB)
        return lab.reshape(3)
    if arr.ndim == 2:  # (N,3) list of colors
        img = arr.reshape(1, -1, 3)
        lab = cv2.cvtColor((img / 255.0).astype(np.float32), cv2.COLOR_RGB2LAB)
        return lab.reshape(-1, 3)
    # (H,W,3) image
    return cv2.cvtColor((arr / 255.0).astype(np.float32), cv2.COLOR_RGB2LAB)


def lab_chroma(lab) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float32)
    a = lab[..., 1]
    b = lab[..., 2]
    return np.sqrt(a * a + b * b)


# --- CLI mini-DSL parsing --------------------------------------------------
def parse_kv(spec: str) -> dict[str, str]:
    """'px=92,val=0,scale=log10' -> {'px':'92','val':'0','scale':'log10'}."""
    out: dict[str, str] = {}
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"expected key=value in {spec!r}, got {token!r}")
        k, v = token.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def parse_xy(spec: str) -> tuple[float, float]:
    """'240,300' or 'px=240,py=300' -> (240.0, 300.0)."""
    if "=" in spec:
        kv = parse_kv(spec)
        return float(kv.get("px", kv.get("x"))), float(kv.get("py", kv.get("y")))
    parts = re.split(r"[,\s]+", spec.strip())
    if len(parts) != 2:
        raise ValueError(f"expected 'x,y', got {spec!r}")
    return float(parts[0]), float(parts[1])


def parse_box(spec: str) -> tuple[int, int, int, int]:
    """'x,y,w,h' -> ints."""
    parts = re.split(r"[,\s]+", spec.strip())
    if len(parts) != 4:
        raise ValueError(f"expected 'x,y,w,h', got {spec!r}")
    return tuple(int(round(float(p))) for p in parts)  # type: ignore[return-value]


def parse_floats(spec: str) -> list[float]:
    return [float(p) for p in re.split(r"[,\s]+", spec.strip()) if p]


# --- JSON IO ---------------------------------------------------------------
class _NpEncoder(json.JSONEncoder):
    def default(self, o: Any):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def write_json(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, cls=_NpEncoder)


def read_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, cls=_NpEncoder)
