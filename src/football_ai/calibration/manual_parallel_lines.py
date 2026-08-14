from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from football_ai.calibration.manual_midfield_line import ManualMidfieldLine


LINE_TYPES = ("midfield", "goal_area_5m", "penalty_area_16m")


@dataclass(frozen=True, slots=True)
class ManualParallelLine:
    line_type: str
    frame_number: int
    time_seconds: float
    points: tuple[tuple[float, float], ...]
    equation: tuple[float, float, float]
    rms_error_px: float
    maximum_error_px: float

    def __post_init__(self) -> None:
        if self.line_type not in LINE_TYPES:
            raise ValueError(f"Onbekend 11v11-lijntype: {self.line_type}")

    @classmethod
    def fit(
        cls,
        line_type: str,
        frame_number: int,
        time_seconds: float,
        points: tuple[tuple[float, float], ...],
    ) -> "ManualParallelLine":
        fitted = ManualMidfieldLine.fit(
            "temporary", frame_number, time_seconds, points
        )
        return cls(
            line_type,
            fitted.frame_number,
            fitted.time_seconds,
            fitted.points,
            fitted.equation,
            fitted.rms_error_px,
            fitted.maximum_error_px,
        )

    @classmethod
    def from_midfield(cls, midfield: ManualMidfieldLine) -> "ManualParallelLine":
        return cls(
            "midfield",
            midfield.frame_number,
            midfield.time_seconds,
            midfield.points,
            midfield.equation,
            midfield.rms_error_px,
            midfield.maximum_error_px,
        )

    def to_dict(self) -> dict:
        return {
            "line_type": self.line_type,
            "frame_number": self.frame_number,
            "time_seconds": self.time_seconds,
            "points": [list(point) for point in self.points],
            "equation": list(self.equation),
            "rms_error_px": self.rms_error_px,
            "maximum_error_px": self.maximum_error_px,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ManualParallelLine":
        return cls(
            str(data["line_type"]),
            int(data["frame_number"]),
            float(data["time_seconds"]),
            tuple(tuple(map(float, point)) for point in data["points"]),
            tuple(map(float, data["equation"])),
            float(data["rms_error_px"]),
            float(data["maximum_error_px"]),
        )


@dataclass(frozen=True, slots=True)
class ManualParallelLineReference:
    video_name: str
    lines: tuple[ManualParallelLine, ...]

    def __post_init__(self) -> None:
        types = tuple(item.line_type for item in self.lines)
        if types != LINE_TYPES:
            raise ValueError("De parallelreferentie vereist middenlijn, 5m-lijn en 16m-lijn.")

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "video_name": self.video_name,
            "world_relation": "parallel",
            "direction_role": "parallel_to_8v8_sidelines",
            "lines": [line.to_dict() for line in self.lines],
        }

    def vanishing_point_at_frame(self, frame_number: int) -> tuple[float, float]:
        lines = tuple(item for item in self.lines if item.frame_number == frame_number)
        if len(lines) < 2:
            raise ValueError("Minimaal twee parallelle lijnen uit hetzelfde frame vereist.")
        equations = np.asarray([item.equation for item in lines], dtype=np.float64)
        _u, _s, vh = np.linalg.svd(equations)
        point = vh[-1]
        if abs(float(point[2])) < 1e-9:
            raise ValueError("Parallelle 11v11-lijnen leveren geen eindig verdwijnpunt.")
        point /= point[2]
        return float(point[0]), float(point[1])

    @classmethod
    def from_dict(cls, data: dict) -> "ManualParallelLineReference":
        return cls(
            str(data["video_name"]),
            tuple(ManualParallelLine.from_dict(line) for line in data["lines"]),
        )


def save_manual_parallel_lines(reference: ManualParallelLineReference, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reference.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def load_manual_parallel_lines(path: Path) -> ManualParallelLineReference:
    return ManualParallelLineReference.from_dict(json.loads(path.read_text(encoding="utf-8")))
