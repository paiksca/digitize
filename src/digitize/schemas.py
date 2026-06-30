"""Serializable description of a digitizing session: manifest, refs, series."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class AxisSpec:
    name: str = ""
    unit: str = ""
    scale: str = "linear"  # linear | log10 | ln | logit
    # for a categorical axis: the ordered category labels (mapped to 0,1,2,...)
    categories: list[str] | None = None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(name=d.get("name", ""), unit=d.get("unit", ""),
                   scale=d.get("scale", "linear"), categories=d.get("categories"))


@dataclass
class SeriesSpec:
    name: str
    color: str | None = None  # hex, e.g. "#1f77b4"; None if shape/style-coded
    kind: str = "scatter"  # scatter | line | bar | errorbar
    marker: str | None = None  # free text, e.g. "circle", "triangle"
    linestyle: str | None = None  # "solid", "dashed", ...
    note: str = ""

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(
            name=d["name"],
            color=d.get("color"),
            kind=d.get("kind", "scatter"),
            marker=d.get("marker"),
            linestyle=d.get("linestyle"),
            note=d.get("note", ""),
        )


@dataclass
class Manifest:
    image: str
    image_size: list[int]  # [width, height]
    plot_type: str = "unknown"
    x: AxisSpec = field(default_factory=AxisSpec)
    y: AxisSpec = field(default_factory=AxisSpec)
    series: list[SeriesSpec] = field(default_factory=list)
    plot_box: list[int] | None = None  # [x, y, w, h]
    notes: str = ""

    def series_by_name(self, name: str) -> SeriesSpec | None:
        for s in self.series:
            if s.name == name:
                return s
        return None

    def to_dict(self):
        return {
            "image": self.image,
            "image_size": list(self.image_size),
            "plot_type": self.plot_type,
            "x": self.x.to_dict(),
            "y": self.y.to_dict(),
            "series": [s.to_dict() for s in self.series],
            "plot_box": self.plot_box,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            image=d["image"],
            image_size=list(d["image_size"]),
            plot_type=d.get("plot_type", "unknown"),
            x=AxisSpec.from_dict(d.get("x", {})),
            y=AxisSpec.from_dict(d.get("y", {})),
            series=[SeriesSpec.from_dict(s) for s in d.get("series", [])],
            plot_box=d.get("plot_box"),
            notes=d.get("notes", ""),
        )


@dataclass
class DataPoint:
    px: float
    py: float
    x: float | None = None
    y: float | None = None
    x_err: float | None = None
    y_err: float | None = None
    # optional secondary error-bar magnitudes (data space), e.g. SD/SEM whiskers
    y_lo: float | None = None
    y_hi: float | None = None
    # pixel-space extras carried from extraction to the values step
    py_hi: float | None = None  # whisker upper row (smaller py = higher value)
    py_lo: float | None = None  # whisker lower row
    area: float | None = None
    score: float | None = None
    # extra pixel-space levels for rich marks (box quartiles, forest CI, bar
    # edges). Keys ending "_px"/"_py" are auto-converted to data by `values`,
    # producing matching "_x"/"_y" keys.
    extra: dict | None = None
    source: str = "auto"  # "auto" | "manual" | "moved"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__ if k in d})


@dataclass
class SeriesData:
    name: str
    kind: str
    color: str | None
    points: list[DataPoint] = field(default_factory=list)

    def to_dict(self):
        return {
            "name": self.name,
            "kind": self.kind,
            "color": self.color,
            "points": [p.to_dict() for p in self.points],
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            name=d["name"],
            kind=d.get("kind", "scatter"),
            color=d.get("color"),
            points=[DataPoint.from_dict(p) for p in d.get("points", [])],
        )
