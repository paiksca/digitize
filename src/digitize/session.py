"""Session state: a directory holding the manifest, calibration, per-series data,
overlays, fits, and an append-only provenance log for reproducibility."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .schemas import Manifest, SeriesData
from .transform import AxisTransform
from .util import read_json, write_json


class Session:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    # ---- locating / creating ---------------------------------------------
    @classmethod
    def for_image(cls, image_path: str | Path, session: str | Path | None = None) -> "Session":
        image_path = Path(image_path)
        root = Path(session) if session else image_path.parent / f"{image_path.stem}.digitize"
        return cls(root)

    @classmethod
    def open(cls, session: str | Path) -> "Session":
        s = cls(session)
        if not s.manifest_path.exists():
            raise FileNotFoundError(
                f"no session at {s.root} (run `digitize init` first)")
        return s

    def ensure(self) -> "Session":
        for d in (self.root, self.overlays_dir, self.series_dir, self.fits_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self

    # ---- paths ------------------------------------------------------------
    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def calibration_path(self) -> Path:
        return self.root / "calibration.json"

    @property
    def overlays_dir(self) -> Path:
        return self.root / "overlays"

    @property
    def series_dir(self) -> Path:
        return self.root / "series"

    @property
    def fits_dir(self) -> Path:
        return self.root / "fits"

    @property
    def provenance_path(self) -> Path:
        return self.root / "provenance.jsonl"

    # ---- manifest ---------------------------------------------------------
    def save_manifest(self, m: Manifest) -> None:
        write_json(self.manifest_path, m.to_dict())

    def load_manifest(self) -> Manifest:
        return Manifest.from_dict(read_json(self.manifest_path))

    # ---- calibration ------------------------------------------------------
    def save_calibration(self, d: dict) -> None:
        write_json(self.calibration_path, d)

    def load_calibration(self) -> dict:
        return read_json(self.calibration_path)

    def has_calibration(self) -> bool:
        return self.calibration_path.exists()

    def transform(self) -> AxisTransform:
        return AxisTransform.from_dict(self.load_calibration()["transform"])

    # ---- series -----------------------------------------------------------
    def _series_file(self, name: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
        return self.series_dir / f"{safe}.json"

    def save_series(self, sd: SeriesData) -> None:
        write_json(self._series_file(sd.name), sd.to_dict())

    def load_series(self, name: str) -> SeriesData:
        return SeriesData.from_dict(read_json(self._series_file(name)))

    def has_series(self, name: str) -> bool:
        return self._series_file(name).exists()

    def list_series(self) -> list[str]:
        if not self.series_dir.exists():
            return []
        return sorted(p.stem for p in self.series_dir.glob("*.json"))

    # ---- tick references (from auto label localization) -------------------
    @property
    def tick_refs_path(self) -> Path:
        return self.root / "tick_refs.json"

    def load_tick_refs(self) -> dict:
        return read_json(self.tick_refs_path) if self.tick_refs_path.exists() else {}

    def save_tick_refs(self, d: dict) -> None:
        write_json(self.tick_refs_path, d)

    @property
    def tick_positions_path(self) -> Path:
        return self.root / "tick_positions.json"

    def load_tick_positions(self) -> dict:
        p = self.tick_positions_path
        return read_json(p) if p.exists() else {}

    def save_tick_positions(self, d: dict) -> None:
        write_json(self.tick_positions_path, d)

    # ---- provenance -------------------------------------------------------
    def log(self, command: str, args: dict, result: dict | None = None) -> None:
        rec = {"ts": datetime.now().isoformat(timespec="seconds"),
               "command": command, "args": args, "result": result or {}}
        self.provenance_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.provenance_path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
